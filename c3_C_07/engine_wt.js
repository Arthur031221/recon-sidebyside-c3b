/* engine.js -- per-frame motion and depth sort for the side-by-side 4D Gaussian viewer.
   The same file runs as a Web Worker (default), on the page's main thread (fallback) and in node (tests).
   Splat texel layout (2 x RGBA32UI per splat, 1024 splats per texture row):
     t0 = [f32 x, f32 y, f32 z, rgba8]   t1 = [half(lsx)|half(lsy), half(lsz)|half(qw), half(qx)|half(qy), half(qz)]
   (ls = log scale, q = unit quaternion w,x,y,z). Static splats sit in slots [0, NsPad), the visible dynamic
   splats of the current frame are packed from slot NsPad on. */
(function (G) {
  'use strict';
  var TW = 1024, MIN_A = 1 / 255, PI = Math.PI, TWO_PI = 2 * Math.PI;

  var H2F = new Float32Array(65536);
  (function () {
    for (var h = 0; h < 65536; h++) {
      var s = (h & 0x8000) ? -1 : 1, e = (h >> 10) & 31, f = h & 1023;
      H2F[h] = s * (e === 0 ? f * 5.960464477539063e-8 : e === 31 ? (f ? NaN : Infinity) : (1 + f / 1024) * Math.pow(2, e - 15));
    }
  })();
  var _f = new Float32Array(1), _u = new Uint32Array(_f.buffer);
  function f2h(v) {
    _f[0] = v;
    var x = _u[0], s = (x >>> 16) & 0x8000, e = ((x >>> 23) & 255) - 112, m = x & 0x7fffff;
    if (e >= 31) return s | 0x7c00;
    if (e <= 0) {
      if (e < -10) return s;
      m = (m | 0x800000) >>> (1 - e);
      return s | ((m + 0x1000) >>> 13);
    }
    return s | ((e << 10) + ((m + 0x1000) >>> 13));
  }

  function parseBin(buf) {
    var L = new DataView(buf).getUint32(0, true);
    var hdr = JSON.parse(new TextDecoder().decode(new Uint8Array(buf, 4, L)));
    var base = 4 + L, out = { meta: hdr.meta || {}, s: {} };
    hdr.sections.forEach(function (s) {
      var n = s.shape.reduce(function (a, b) { return a * b; }, 1), off = base + s.offset, a;
      if (s.dtype === 'float32') a = new Float32Array(buf, off, n);
      else if (s.dtype === 'float16' || s.dtype === 'uint16') a = new Uint16Array(buf, off, n);
      else if (s.dtype === 'uint8') a = new Uint8Array(buf, off, n);
      else if (s.dtype === 'int32') a = new Int32Array(buf, off, n);
      else if (s.dtype === 'uint32') a = new Uint32Array(buf, off, n);
      else throw new Error('dtype ' + s.dtype + ' in section ' + s.name);
      out.s[s.name] = { a: a, shape: s.shape, dtype: s.dtype };
    });
    return out;
  }
  function sec(B, name, opt) {
    var s = B.s[name];
    if (!s) { if (opt) return null; throw new Error('missing section ' + name); }
    return s.a;
  }
  function halfToF32(a) { var o = new Float32Array(a.length); for (var i = 0; i < a.length; i++) o[i] = H2F[a[i]]; return o; }

  // static splat texels (2 x RGBA32UI per splat) padded to NsPad slots
  function staticTexels(S, NsPad) {
    var Ns = S.meta.N, i, b;
    var pos = sec(S, 'pos'), ls = sec(S, 'ls'), q = sec(S, 'quat'), rgba = sec(S, 'rgba');
    var st = new Uint32Array(NsPad * 8), stF = new Float32Array(st.buffer);
    for (i = 0; i < Ns; i++) {
      b = i * 8;
      stF[b] = pos[3 * i]; stF[b + 1] = pos[3 * i + 1]; stF[b + 2] = pos[3 * i + 2];
      st[b + 3] = rgba[4 * i] | (rgba[4 * i + 1] << 8) | (rgba[4 * i + 2] << 16) | (rgba[4 * i + 3] << 24);
      st[b + 4] = ls[3 * i] | (ls[3 * i + 1] << 16);
      st[b + 5] = ls[3 * i + 2] | (q[4 * i] << 16);
      st[b + 6] = q[4 * i + 1] | (q[4 * i + 2] << 16);
      st[b + 7] = q[4 * i + 3];
    }
    return { st: st, pos: new Float32Array(pos), N: Ns };
  }

  function Model(sBuf, dBuf) {
    var S = sBuf ? parseBin(sBuf) : null, D = parseBin(dBuf), i;
    this.kind = D.meta.kind;
    var Ns = S ? S.meta.N : 0, NsPad = Math.ceil(Ns / TW) * TW, Nd = D.meta.N;
    this.Ns = Ns; this.NsPad = NsPad; this.Nd = Nd; this.T = D.meta.T;
    this.rows = NsPad / TW + Math.ceil(Nd / TW);
    // ---- static texels (built once); a dynamic-only model (WorldTree members) passes sBuf = null
    if (S) { var sr = staticTexels(S, NsPad); this.sPos = sr.pos; this.stat = sr.st; }
    // ---- dynamic: constant parts
    var dls = sec(D, 'ls'), drgba = sec(D, 'rgba');
    this.dLs01 = new Uint32Array(Nd); this.dLs2 = new Uint32Array(Nd); this.dRgb = new Uint32Array(Nd); this.dA = new Uint8Array(Nd);
    for (i = 0; i < Nd; i++) {
      this.dLs01[i] = dls[3 * i] | (dls[3 * i + 1] << 16);
      this.dLs2[i] = dls[3 * i + 2];
      this.dRgb[i] = drgba[4 * i] | (drgba[4 * i + 1] << 8) | (drgba[4 * i + 2] << 16);
      this.dA[i] = drgba[4 * i + 3];
    }
    // per-frame outputs, compacted to the visible dynamic splats (k < V); dIdx[k] = source index
    this.dPos = new Float32Array(Nd * 3); this.dQ = new Float32Array(Nd * 4); this.dO = new Float32Array(Nd);
    this.dIdx = new Uint32Array(Nd); this.V = 0; this.j = -1;
    var cx = 0, cy = 0, cz = 0, src;
    if (this.kind === 'mosca') {
      var M = D.meta.M, K = D.meta.K;
      this.mo = {
        K: K, M: M, x: new Float32Array(sec(D, 'x')), qs: new Float32Array(sec(D, 'qsrc')),
        ski: new Uint16Array(sec(D, 'sk_ind')), skw: new Float32Array(sec(D, 'sk_w')), dyo: new Float32Array(sec(D, 'dyn_o')),
        rt: new Uint16Array(sec(D, 'ref_time')), nx: new Float32Array(sec(D, 'node_xyz')), nq: new Float32Array(sec(D, 'node_quat')),
        cache: new Float32Array(this.T * M * 8), stamp: new Int32Array(this.T * M), st: 0
      };
      src = this.mo.x;
    } else if (this.kind === 'm8') {
      var md = D.meta.m8d || null, on = !!(md && md.delta_on);
      var ss = sec(D, 'ss'), invS = new Float32Array(Nd);
      for (i = 0; i < Nd; i++) invS[i] = Math.exp(-ss[i]);
      var nod = on && md.has_nod ? new Float32Array(sec(D, 'nod')) : null;
      var nodes = nod ? new Float32Array(sec(D, 'nodes')) : null;
      this.m8 = {
        mean: new Float32Array(sec(D, 'mean')), vel: new Float32Array(sec(D, 'vel')), sc: new Float32Array(sec(D, 'sc')),
        invS: invS, q: halfToF32(sec(D, 'quat')), op: new Float32Array(sec(D, 'op')), U: new Float32Array(sec(D, 'U')),
        periodic: !!D.meta.periodic, on: on,
        rig: on && md.has_rig ? new Float32Array(sec(D, 'rig')) : null,
        nod: nod, nodes: nodes, Nn: nodes ? nodes.length / 3 : 0,
        ph: on && md.has_ph ? new Float32Array(sec(D, 'ph')) : null,
        knn: md ? md.knn : 0, sig: md ? md.sig_n : 1, rb: md ? md.rb : 1, X0: md ? md.X0 : [0, 0, 0]
      };
      src = this.m8.mean;
    } else throw new Error('unknown model kind ' + this.kind);
    for (i = 0; i < Nd; i++) { cx += src[3 * i]; cy += src[3 * i + 1]; cz += src[3 * i + 2]; }
    this.center = Nd ? [cx / Nd, cy / Nd, cz / Nd] : [0, 0, 0];
  }

  Model.prototype.info = function () {
    return { kind: this.kind, Ns: this.Ns, NsPad: this.NsPad, Nd: this.Nd, rows: this.rows, T: this.T, center: this.center };
  };

  // ---- MoSca: dual-quaternion blending of the scaffold nodes (node pose at t relative to the splat's reference time)
  Model.prototype._nodeDQ = function (t, r, n, out, o8) {
    var mo = this.mo, M = mo.M, nq = mo.nq, nx = mo.nx;
    var a = (t * M + n) * 4, c = (r * M + n) * 4;
    var aw = nq[a], ax = nq[a + 1], ay = nq[a + 2], az = nq[a + 3];
    var bw = nq[c], bx = -nq[c + 1], by = -nq[c + 2], bz = -nq[c + 3];          // conj(q_ref)
    var pw = aw * bw - ax * bx - ay * by - az * bz, px = aw * bx + ax * bw + ay * bz - az * by,
      py = aw * by - ax * bz + ay * bw + az * bx, pz = aw * bz + ax * by - ay * bx + az * bw;
    var nn = Math.sqrt(pw * pw + px * px + py * py + pz * pz) || 1;
    if (pw < 0) nn = -nn;                                                          // standardize (w >= 0)
    pw /= nn; px /= nn; py /= nn; pz /= nn;
    var xr = (r * M + n) * 3, xt = (t * M + n) * 3, vx = nx[xr], vy = nx[xr + 1], vz = nx[xr + 2];
    var cx1 = py * vz - pz * vy, cy1 = pz * vx - px * vz, cz1 = px * vy - py * vx;   // rotate v by p
    var rx = vx + 2 * (pw * cx1 + py * cz1 - pz * cy1), ry = vy + 2 * (pw * cy1 + pz * cx1 - px * cz1),
      rz = vz + 2 * (pw * cz1 + px * cy1 - py * cx1);
    var tx = nx[xt] - rx, ty = nx[xt + 1] - ry, tz = nx[xt + 2] - rz;
    out[o8] = pw; out[o8 + 1] = px; out[o8 + 2] = py; out[o8 + 3] = pz;
    out[o8 + 4] = 0.5 * (-tx * px - ty * py - tz * pz); out[o8 + 5] = 0.5 * (tx * pw + ty * pz - tz * py);
    out[o8 + 6] = 0.5 * (-tx * pz + ty * pw + tz * px); out[o8 + 7] = 0.5 * (tx * py - ty * px + tz * pw);
  };

  Model.prototype._moscaFrame = function (t) {
    var mo = this.mo, K = mo.K, M = mo.M, N = this.Nd, x = mo.x, qs = mo.qs, ski = mo.ski, skw = mo.skw, dyo = mo.dyo, rt = mo.rt;
    var cache = mo.cache, stamp = mo.stamp, st = ++mo.st;
    var P = this.dPos, Q = this.dQ, O = this.dO, I = this.dIdx, A = this.dA, V = 0;
    for (var i = 0; i < N; i++) {
      if (A[i] === 0) continue;
      var r = rt[i], rb = r * M;
      var pw = 0, px = 0, py = 0, pz = 0, qw = 0, qx = 0, qy = 0, qz = 0;
      for (var k = 0, ik = i * K; k < K; k++, ik++) {
        var w = skw[ik], n = ski[ik];
        if (w === 0 || n >= M) continue;
        var ci = rb + n, c8 = ci * 8;
        if (stamp[ci] !== st) { this._nodeDQ(t, r, n, cache, c8); stamp[ci] = st; }
        pw += w * cache[c8]; px += w * cache[c8 + 1]; py += w * cache[c8 + 2]; pz += w * cache[c8 + 3];
        qw += w * cache[c8 + 4]; qx += w * cache[c8 + 5]; qy += w * cache[c8 + 6]; qz += w * cache[c8 + 7];
      }
      var dd = dyo[i];
      pw = pw * dd + (1 - dd); px *= dd; py *= dd; pz *= dd; qw *= dd; qx *= dd; qy *= dd; qz *= dd;
      var rr = Math.sqrt(pw * pw + px * px + py * py + pz * pz);
      if (rr < 1e-4) { pw = 1; px = py = pz = 0; qw = qx = qy = qz = 0; }
      else {
        var ir = 1 / rr; pw *= ir; px *= ir; py *= ir; pz *= ir; qw *= ir; qx *= ir; qy *= ir; qz *= ir;
        var dp = pw * qw + px * qx + py * qy + pz * qz; qw -= pw * dp; qx -= px * dp; qy -= py * dp; qz -= pz * dp;
      }
      // translation = 2 (q_d * conj(p)).xyz
      var tx = 2 * (-qw * px + qx * pw - qy * pz + qz * py), ty = 2 * (-qw * py + qx * pz + qy * pw - qz * px),
        tz = 2 * (-qw * pz - qx * py + qy * px + qz * pw);
      var vx = x[3 * i], vy = x[3 * i + 1], vz = x[3 * i + 2];
      var cx1 = py * vz - pz * vy, cy1 = pz * vx - px * vz, cz1 = px * vy - py * vx;
      var b3 = V * 3, b4 = V * 4;
      P[b3] = vx + 2 * (pw * cx1 + py * cz1 - pz * cy1) + tx;
      P[b3 + 1] = vy + 2 * (pw * cy1 + pz * cx1 - px * cz1) + ty;
      P[b3 + 2] = vz + 2 * (pw * cz1 + px * cy1 - py * cx1) + tz;
      var bw = qs[4 * i], bx = qs[4 * i + 1], by = qs[4 * i + 2], bz = qs[4 * i + 3];
      Q[b4] = pw * bw - px * bx - py * by - pz * bz; Q[b4 + 1] = pw * bx + px * bw + py * bz - pz * by;
      Q[b4 + 2] = pw * by - px * bz + py * bw + pz * bx; Q[b4 + 3] = pw * bz + px * by - py * bx + pz * bw;
      O[V] = A[i] / 255; I[V] = i; V++;
    }
    this.V = V;
  };

  // ---- method 8 (m8): state-sliced splats, optional per-frame node field + rigid motion
  function so3exp(wx, wy, wz) {
    var th2 = wx * wx + wy * wy + wz * wz, th = Math.sqrt(Math.max(th2, 1e-24)), small = th2 < 1e-8;
    var A = small ? 1 - th2 / 6 : Math.sin(th) / th, B = small ? 0.5 - th2 / 24 : (1 - Math.cos(th)) / Math.max(th2, 1e-24);
    // R = I + A K + B K^2, K^2 = w w^T - th2 I
    return [1 + B * (wx * wx - th2), -A * wz + B * wx * wy, A * wy + B * wx * wz,
      A * wz + B * wx * wy, 1 + B * (wy * wy - th2), -A * wx + B * wy * wz,
      -A * wy + B * wx * wz, A * wx + B * wy * wz, 1 + B * (wz * wz - th2)];
  }
  function aa2q(wx, wy, wz) {
    var th = Math.sqrt(Math.max(wx * wx + wy * wy + wz * wz, 1e-24));
    var s = th < 1e-6 ? 0.5 - th * th / 48 : Math.sin(0.5 * th) / th;
    return [Math.cos(0.5 * th), s * wx, s * wy, s * wz];
  }

  Model.prototype._m8Frame = function (j) {
    var o = this.m8, N = this.Nd, P = this.dPos, Q = this.dQ, O = this.dO, I = this.dIdx;
    var mean = o.mean, vel = o.vel, sc = o.sc, invS = o.invS, op = o.op, q = o.q;
    var u = o.U[j] + (o.on && o.ph ? o.ph[j] : 0), periodic = o.periodic;
    var useNod = !!(o.on && o.nod), useRig = !!(o.on && o.rig);
    var R = null, qa = null, X0 = o.X0, trx = 0, trY = 0, trz = 0;
    if (useRig) {
      var b6 = j * 6, rb = o.rb, wx = o.rig[b6] / rb, wy = o.rig[b6 + 1] / rb, wz = o.rig[b6 + 2] / rb;
      R = so3exp(wx, wy, wz); qa = aa2q(wx, wy, wz); trx = o.rig[b6 + 3]; trY = o.rig[b6 + 4]; trz = o.rig[b6 + 5];
    }
    var Nn = o.Nn, nodes = o.nodes, nod = o.nod, nb = j * Nn * 3, kn = o.knn, isg2 = 1 / (o.sig * o.sig);
    var bd = new Float64Array(Math.max(kn, 1)), bi = new Int32Array(Math.max(kn, 1)), ew = new Float64Array(Math.max(kn, 1));
    var V = 0;
    for (var i = 0; i < N; i++) {
      var d = u - sc[i];
      if (periodic) { d += PI; d -= TWO_PI * Math.floor(d / TWO_PI); d -= PI; }
      var g = d * invS[i], a = op[i] * Math.exp(-0.5 * g * g);
      if (a < MIN_A) continue;
      var x = mean[3 * i] + vel[3 * i] * d, y = mean[3 * i + 1] + vel[3 * i + 1] * d, z = mean[3 * i + 2] + vel[3 * i + 2] * d;
      var qw = q[4 * i], qx = q[4 * i + 1], qy = q[4 * i + 2], qz = q[4 * i + 3], kk, mm;
      if (useNod) {
        for (kk = 0; kk < kn; kk++) { bd[kk] = Infinity; bi[kk] = 0; }
        for (mm = 0; mm < Nn; mm++) {
          var ex = x - nodes[3 * mm], ey = y - nodes[3 * mm + 1], ez = z - nodes[3 * mm + 2], d2 = ex * ex + ey * ey + ez * ez;
          if (d2 < bd[kn - 1]) {
            var p2 = kn - 1;
            while (p2 > 0 && bd[p2 - 1] > d2) { bd[p2] = bd[p2 - 1]; bi[p2] = bi[p2 - 1]; p2--; }
            bd[p2] = d2; bi[p2] = mm;
          }
        }
        var mx = -Infinity, sw = 0;
        for (kk = 0; kk < kn; kk++) { ew[kk] = -0.5 * bd[kk] * isg2; if (ew[kk] > mx) mx = ew[kk]; }
        for (kk = 0; kk < kn; kk++) { ew[kk] = Math.exp(ew[kk] - mx); sw += ew[kk]; }
        var ox = 0, oy = 0, oz = 0;
        for (kk = 0; kk < kn; kk++) {
          var wk = ew[kk] / sw, nb3 = nb + bi[kk] * 3;
          ox += wk * nod[nb3]; oy += wk * nod[nb3 + 1]; oz += wk * nod[nb3 + 2];
        }
        x += ox; y += oy; z += oz;
      }
      if (useRig) {
        var lx = x - X0[0], ly = y - X0[1], lz = z - X0[2];
        x = R[0] * lx + R[1] * ly + R[2] * lz + X0[0] + trx;
        y = R[3] * lx + R[4] * ly + R[5] * lz + X0[1] + trY;
        z = R[6] * lx + R[7] * ly + R[8] * lz + X0[2] + trz;
        var aw = qa[0], ax = qa[1], ay = qa[2], az = qa[3];
        var nw = aw * qw - ax * qx - ay * qy - az * qz, nx2 = aw * qx + ax * qw + ay * qz - az * qy,
          ny = aw * qy - ax * qz + ay * qw + az * qx, nz = aw * qz + ax * qy - ay * qx + az * qw;
        qw = nw; qx = nx2; qy = ny; qz = nz;
      }
      var b3 = V * 3, b4 = V * 4;
      P[b3] = x; P[b3 + 1] = y; P[b3 + 2] = z;
      Q[b4] = qw; Q[b4 + 1] = qx; Q[b4 + 2] = qy; Q[b4 + 3] = qz;
      O[V] = a; I[V] = i; V++;
    }
    this.V = V;
  };

  Model.prototype.frame = function (j) {
    j = Math.max(0, Math.min(this.T - 1, j | 0));
    if (j !== this.j) { if (this.kind === 'mosca') this._moscaFrame(j); else this._m8Frame(j); this.j = j; return true; }
    return false;
  };

  Model.prototype.packDyn = function () {
    var V = this.V, out = new Uint32Array(Math.ceil(V / TW) * TW * 8), F = new Float32Array(out.buffer);
    var P = this.dPos, Q = this.dQ, O = this.dO, I = this.dIdx;
    for (var k = 0; k < V; k++) {
      var i = I[k], b = k * 8, qw = Q[4 * k], qx = Q[4 * k + 1], qy = Q[4 * k + 2], qz = Q[4 * k + 3];
      var nn = Math.sqrt(qw * qw + qx * qx + qy * qy + qz * qz) || 1;
      var a = Math.min(255, Math.max(1, Math.round(O[k] * 255)));
      F[b] = P[3 * k]; F[b + 1] = P[3 * k + 1]; F[b + 2] = P[3 * k + 2];
      out[b + 3] = this.dRgb[i] | (a << 24);
      out[b + 4] = this.dLs01[i];
      out[b + 5] = this.dLs2[i] | (f2h(qw / nn) << 16);
      out[b + 6] = f2h(qx / nn) | (f2h(qy / nn) << 16);
      out[b + 7] = f2h(qz / nn);
    }
    return out;
  };

  // back-to-front order of the splat slots for one camera (view = row-major 4x4 world-to-camera, OpenCV axes)
  Model.prototype.sort = function (view, tanx, tany) {
    var cap = this.Ns + this.Nd;
    if (!this._z) { this._z = new Float32Array(cap); this._s = new Uint32Array(cap); this._k = new Uint16Array(cap); this._cnt = new Uint32Array(65536); }
    var Z = this._z, S = this._s, KY = this._k, cnt = this._cnt;
    var a0 = view[0], a1 = view[1], a2 = view[2], a3 = view[3], b0 = view[4], b1 = view[5], b2 = view[6], b3 = view[7],
      c0 = view[8], c1 = view[9], c2 = view[10], c3 = view[11];
    var lx = 2 * tanx, ly = 2 * tany, m = 0, zmin = Infinity, zmax = -Infinity, i, x, y, z, zc, xc, yc;
    var src = this.sPos, n = this.Ns, off = 0;
    for (var pass = 0; pass < 2; pass++) {
      if (pass === 1) { src = this.dPos; n = this.V; off = this.NsPad; }
      for (i = 0; i < n; i++) {
        x = src[3 * i]; y = src[3 * i + 1]; z = src[3 * i + 2];
        zc = c0 * x + c1 * y + c2 * z + c3;
        if (zc < 0.02) continue;
        xc = a0 * x + a1 * y + a2 * z + a3; if (xc > zc * lx || -xc > zc * lx) continue;
        yc = b0 * x + b1 * y + b2 * z + b3; if (yc > zc * ly || -yc > zc * ly) continue;
        Z[m] = zc; S[m] = off + i; m++;
        if (zc < zmin) zmin = zc; if (zc > zmax) zmax = zc;
      }
    }
    var span = zmax - zmin, s = span > 0 ? 65535 / span : 0, key, acc = 0, c;
    cnt.fill(0);
    for (i = 0; i < m; i++) { key = 65535 - (((Z[i] - zmin) * s) | 0); KY[i] = key; cnt[key]++; }
    for (i = 0; i < 65536; i++) { c = cnt[i]; cnt[i] = acc; acc += c; }
    var order = new Uint32Array(m);
    for (i = 0; i < m; i++) order[cnt[KY[i]]++] = S[i];
    return order;
  };

  Model.prototype.update = function (j, view, tanx, tany) {
    var changed = this.frame(j);
    var dyn = changed ? this.packDyn() : null;
    var order = this.sort(view, tanx, tany);
    return { j: this.j, V: this.V, dyn: dyn, order: order, n: order.length };
  };


  // ---- WorldTree (height 2): 4 leaves with disjoint time windows; frame tid uses the leaf holding tid (hard switch).
  // Gaussians = that leaf's own static + dynamic members [leaf, parent, root] (per-leaf nested copies, no ref-time mask),
  // each member queried at local time tid - member_left, all drawn together in one pass.
  // meta = { leaves: [{lo, hi, members: [{left}, {left}, {left}]}] }; statics: 4 buffers; dyns: [4][3] buffers.
  function WTModel(sBufs, dBufs, meta) {
    var self = this, k, j;
    this.kind = 'wt'; this.meta = meta; this.nL = meta.leaves.length;
    this.lo = meta.leaves.map(function (l) { return l.lo; }); this.hi = meta.leaves.map(function (l) { return l.hi; });
    this.T = this.hi[this.nL - 1] + 1;
    var S = sBufs.map(parseBin), NsPad = 0, maxNs = 0, maxNd = 0;
    S.forEach(function (s) { maxNs = Math.max(maxNs, s.meta.N); });
    NsPad = Math.ceil(maxNs / TW) * TW;
    this.leafStat = []; this.leafPos = []; this.leafNs = [];
    for (k = 0; k < this.nL; k++) {
      var sr = staticTexels(S[k], NsPad);
      this.leafStat.push(sr.st); this.leafPos.push(sr.pos); this.leafNs.push(sr.N);
    }
    S = null;
    this.mem = []; this.left = [];
    for (k = 0; k < this.nL; k++) {
      var ms = [], lf = [], nd = 0;
      for (j = 0; j < dBufs[k].length; j++) { var m = new Model(null, dBufs[k][j]); ms.push(m); lf.push(meta.leaves[k].members[j].left); nd += m.Nd; }
      this.mem.push(ms); this.left.push(lf); maxNd = Math.max(maxNd, nd);
    }
    this.NsPad = NsPad; this.Ns = maxNs; this.Nd = maxNd;
    this.rows = NsPad / TW + Math.ceil(maxNd / TW);
    this.dPos = new Float32Array(maxNd * 3); this.V = 0; this.j = -1; this.k = -1;
    this.sPos = this.leafPos[0];
    var c = [0, 0, 0], n = 0;   // camera target: centre of the root model of the first leaf
    var rc = this.mem[0][this.mem[0].length - 1].center; c = rc;
    this.center = c;
    this.stat = this.leafStat[0];   // static texels of leaf 0 (the worker sends a copy at load)
  }
  WTModel.prototype.info = function () {
    return { kind: 'wt', Ns: this.Ns, NsPad: this.NsPad, Nd: this.Nd, rows: this.rows, T: this.T, center: this.center };
  };
  WTModel.prototype.leafOf = function (tid) {
    for (var k = 0; k < this.nL; k++) if (tid <= this.hi[k]) return k;
    return this.nL - 1;
  };
  WTModel.prototype.frame = function (tid) {
    tid = Math.max(0, Math.min(this.T - 1, tid | 0));
    if (tid === this.j) return false;
    var k = this.leafOf(tid), ms = this.mem[k], o = 0;
    for (var j = 0; j < ms.length; j++) {
      var m = ms[j]; m.frame(tid - this.left[k][j]);
      this.dPos.set(m.dPos.subarray(0, 3 * m.V), 3 * o); o += m.V;
    }
    this.V = o; this.j = tid;
    this.statChanged = (k !== this.k);
    this.k = k; this.sPos = this.leafPos[k]; this.Ns = this.leafNs[k]; this.sk = k;
    return true;
  };
  WTModel.prototype.packDyn = function () {
    var out = new Uint32Array(Math.ceil(this.V / TW) * TW * 8), o = 0, ms = this.mem[this.k];
    for (var j = 0; j < ms.length; j++) {
      var m = ms[j]; if (!m.V) continue;
      out.set(m.packDyn().subarray(0, m.V * 8), o * 8); o += m.V;
    }
    return out;
  };
  WTModel.prototype.sort = Model.prototype.sort;
  WTModel.prototype.update = function (tid, view, tanx, tany) {
    var changed = this.frame(tid), stat = null;
    if (changed && this.statChanged) { stat = this.leafStat[this.k].slice(); this.statChanged = false; }
    var dyn = changed ? this.packDyn() : null;
    if (!this._z) { var cap = this.NsPad + this.Nd; this._z = new Float32Array(cap); this._s = new Uint32Array(cap); this._k = new Uint16Array(cap); this._cnt = new Uint32Array(65536); }
    var order = this.sort(view, tanx, tany);
    return { j: this.j, V: this.V, dyn: dyn, stat: stat, order: order, n: order.length };
  };

  var api = { Model: Model, WTModel: WTModel, staticTexels: staticTexels, parseBin: parseBin, H2F: H2F, f2h: f2h, TW: TW };
  G.V3D = api;
  if (typeof module !== 'undefined' && module.exports) module.exports = api;

  // ---- worker glue
  var isWorker = typeof WorkerGlobalScope !== 'undefined' && typeof self !== 'undefined' && self instanceof WorkerGlobalScope;
  if (isWorker) {
    var model = null;
    self.onmessage = function (ev) {
      var d = ev.data;
      try {
        if (d.type === 'load') {
          if (d.wt) { model = new WTModel(d.s, d.d, d.wt); var stat0 = model.leafStat[0].slice(); self.postMessage({ type: 'loaded', info: model.info(), stat: stat0 }, [stat0.buffer]); }
          else {
            model = new Model(d.s, d.d);
            var stat = model.stat; model.stat = null;
            self.postMessage({ type: 'loaded', info: model.info(), stat: stat }, [stat.buffer]);
          }
        } else if (d.type === 'update' && model) {
          var r = model.update(d.j, d.view, d.tanx, d.tany);
          r.type = 'frame'; r.id = d.id;
          var tr = [r.order.buffer]; if (r.dyn) tr.push(r.dyn.buffer); if (r.stat) tr.push(r.stat.buffer);
          self.postMessage(r, tr);
        }
      } catch (e) {
        self.postMessage({ type: 'error', msg: String((e && e.stack) || e) });
      }
    };
    self.postMessage({ type: 'ready' });
  }
})(typeof globalThis !== 'undefined' ? globalThis : this);
