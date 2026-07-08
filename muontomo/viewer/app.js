// Muon tomography volume viewer. No modules, no fetch -- everything is inline.
// window.__VOLUME__ is injected by build.py before this script runs:
//   { shape:[nx,ny,nz], origin_m:[x0,y0,z0], spacing_m, value_range:[lo,hi],
//     suggested_iso:[a,b], run, headline_metrics:{...}, data_b64 (uint8, row-major x,y,z) }
(function () {
  "use strict";
  const V = window.__VOLUME__;
  const [nx, ny, nz] = V.shape;

  function decodeVolume(b64) {
    const bin = atob(b64);
    const arr = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) arr[i] = bin.charCodeAt(i);
    return arr;
  }
  const field = decodeVolume(V.data_b64); // uint8, 0..255 maps to [0, value_range[1]]

  // ---- three.js scene ----
  const container = document.getElementById("viewport");
  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x101418);
  const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 1000);
  const renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setPixelRatio(window.devicePixelRatio || 1);
  container.appendChild(renderer.domElement);

  const controls = new THREE.OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;

  scene.add(new THREE.AmbientLight(0xffffff, 0.6));
  const dir = new THREE.DirectionalLight(0xffffff, 0.8);
  dir.position.set(1, 1.5, 1);
  scene.add(dir);

  const spacing = V.spacing_m;
  const origin = V.origin_m; // world (x, y, z) with z vertical (ceiling direction)
  // Three.js/OrbitControls assume Y is up; the volume's vertical axis is world Z.
  // toThree(x, y, z) remaps world -> scene axes consistently everywhere below.
  const toThree = (x, y, z) => new THREE.Vector3(x, z, y);
  const originT = toThree(...origin);
  const center = toThree(
    origin[0] + (nx * spacing) / 2,
    origin[1] + (ny * spacing) / 2,
    origin[2] + (nz * spacing) / 2
  );
  const diag = Math.hypot(nx * spacing, ny * spacing, nz * spacing);

  const grid = new THREE.GridHelper(Math.max(nx, ny) * spacing, 10, 0x445, 0x334);
  grid.position.set(center.x, originT.y, center.z);
  scene.add(grid);
  const axes = new THREE.AxesHelper(diag * 0.15);
  axes.position.copy(originT);
  scene.add(axes);

  const group = new THREE.Group();
  scene.add(group);
  let meshLo = null, meshHi = null;

  function toWorld(geom) {
    const pos = geom.attributes.position;
    for (let i = 0; i < pos.count; i++) {
      const p = toThree(
        origin[0] + pos.getX(i) * spacing,
        origin[1] + pos.getY(i) * spacing,
        origin[2] + pos.getZ(i) * spacing
      );
      pos.setXYZ(i, p.x, p.y, p.z);
    }
    geom.computeVertexNormals();
    return geom;
  }

  function buildMesh(isoFrac, color, opacity) {
    const iso = isoFrac * 255;
    const verts = window.marchingCubes(field, nx, ny, nz, iso);
    const geom = new THREE.BufferGeometry();
    geom.setAttribute("position", new THREE.BufferAttribute(verts, 3));
    toWorld(geom);
    const mat = new THREE.MeshStandardMaterial({
      color, opacity, transparent: opacity < 1, side: THREE.DoubleSide, roughness: 0.7,
    });
    return new THREE.Mesh(geom, mat);
  }

  const state = {
    threshold: (V.suggested_iso[0] || 20) / 255,
    threshold2: (V.suggested_iso[1] || 60) / 255,
    showHigh: true,
    zSlice: nz - 1,
    clipBelow: false,
  };

  function rebuild() {
    if (meshLo) { group.remove(meshLo); meshLo.geometry.dispose(); meshLo.material.dispose(); }
    if (meshHi) { group.remove(meshHi); meshHi.geometry.dispose(); meshHi.material.dispose(); }
    meshLo = buildMesh(state.threshold, 0x4fa3ff, 0.35);
    group.add(meshLo);
    if (state.showHigh) {
      meshHi = buildMesh(state.threshold2, 0xff8c42, 0.9);
      group.add(meshHi);
    } else {
      meshHi = null;
    }
    if (state.clipBelow) {
      const yw = origin[2] + state.zSlice * spacing; // three-space Y = world Z (vertical)
      [meshLo, meshHi].forEach((m) => {
        if (!m) return;
        m.material.clippingPlanes = [new THREE.Plane(new THREE.Vector3(0, 1, 0), -yw)];
      });
      renderer.localClippingEnabled = true;
    } else {
      [meshLo, meshHi].forEach((m) => m && (m.material.clippingPlanes = []));
    }
    lastTriangles = (meshLo ? meshLo.geometry.attributes.position.count / 3 : 0) +
                    (meshHi ? meshHi.geometry.attributes.position.count / 3 : 0);
  }
  let lastTriangles = 0;

  // z-slice plane, textured with a viridis LUT of the slab at zSlice
  const VIRIDIS = [[68,1,84],[59,82,139],[33,145,140],[94,201,98],[253,231,37]];
  function viridis(t) {
    t = Math.max(0, Math.min(1, t));
    const n = VIRIDIS.length - 1;
    const i = Math.min(n - 1, Math.floor(t * n));
    const f = t * n - i;
    const a = VIRIDIS[i], b = VIRIDIS[i + 1];
    return [a[0] + f * (b[0] - a[0]), a[1] + f * (b[1] - a[1]), a[2] + f * (b[2] - a[2])];
  }
  const sliceCanvas = document.createElement("canvas");
  sliceCanvas.width = nx; sliceCanvas.height = ny;
  const sliceCtx = sliceCanvas.getContext("2d");
  const sliceTexture = new THREE.CanvasTexture(sliceCanvas);
  const sliceGeom = new THREE.PlaneGeometry(nx * spacing, ny * spacing);
  const sliceMat = new THREE.MeshBasicMaterial({ map: sliceTexture, transparent: true, opacity: 0.9, side: THREE.DoubleSide });
  const sliceMesh = new THREE.Mesh(sliceGeom, sliceMat);
  sliceMesh.rotation.x = -Math.PI / 2;
  scene.add(sliceMesh);
  let sliceVisible = true;

  function updateSlice() {
    const iz = Math.max(0, Math.min(nz - 1, state.zSlice));
    const img = sliceCtx.createImageData(nx, ny);
    let vmax = 1;
    for (let x = 0; x < nx; x++) for (let y = 0; y < ny; y++) {
      vmax = Math.max(vmax, field[(x * ny + y) * nz + iz]);
    }
    for (let x = 0; x < nx; x++) {
      for (let y = 0; y < ny; y++) {
        const v = field[(x * ny + y) * nz + iz] / vmax;
        const [r, g, b] = viridis(v);
        const idx = ((ny - 1 - y) * nx + x) * 4;
        img.data[idx] = r; img.data[idx + 1] = g; img.data[idx + 2] = b;
        img.data[idx + 3] = v < 0.02 ? 0 : 220;
      }
    }
    sliceCtx.putImageData(img, 0, 0);
    sliceTexture.needsUpdate = true;
    sliceMesh.position.set(center.x, origin[2] + iz * spacing, center.z);
    sliceMesh.visible = sliceVisible;
  }

  rebuild();
  updateSlice();

  function frameCamera(kind) {
    // Three-space Y is always up here (world Z), so a single up vector suffices.
    const R = diag * 1.3;
    camera.up.set(0, 1, 0);
    if (kind === "top") camera.position.set(center.x, originT.y + R, center.z + 0.001);
    else if (kind === "front") camera.position.set(center.x, center.y, originT.z - R);
    else camera.position.set(center.x + R * 0.6, originT.y + R * 0.5, center.z + R * 0.6);
    camera.lookAt(center.x, center.y, center.z);
    controls.target.copy(center);
    controls.update();
  }
  frameCamera("iso");

  function resize() {
    const w = container.clientWidth, h = container.clientHeight;
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
    renderer.setSize(w, h);
  }
  window.addEventListener("resize", resize);
  resize();

  function animate() {
    requestAnimationFrame(animate);
    controls.update();
    renderer.render(scene, camera);
  }
  animate();

  // ---- UI wiring ----
  function $(id) { return document.getElementById(id); }
  $("hud-title").textContent = `run: ${V.run}`;
  $("hud-metrics").textContent = Object.entries(V.headline_metrics || {})
    .map(([k, v]) => `${k}: ${typeof v === "number" ? v.toPrecision(3) : v}`)
    .join("  |  ");

  $("thresh-lo").addEventListener("input", (e) => { state.threshold = +e.target.value; rebuild(); });
  $("thresh-hi").addEventListener("input", (e) => { state.threshold2 = +e.target.value; rebuild(); });
  $("toggle-hi").addEventListener("change", (e) => { state.showHigh = e.target.checked; rebuild(); });
  $("zslice").addEventListener("input", (e) => { state.zSlice = +e.target.value; updateSlice(); if (state.clipBelow) rebuild(); });
  $("toggle-slice").addEventListener("change", (e) => { sliceVisible = e.target.checked; sliceMesh.visible = sliceVisible; });
  $("toggle-clip").addEventListener("change", (e) => { state.clipBelow = e.target.checked; rebuild(); });
  $("cam-iso").addEventListener("click", () => frameCamera("iso"));
  $("cam-top").addEventListener("click", () => frameCamera("top"));
  $("cam-front").addEventListener("click", () => frameCamera("front"));

  const zslider = $("zslice");
  zslider.max = nz - 1;
  zslider.value = state.zSlice;
  $("thresh-lo").value = state.threshold;
  $("thresh-hi").value = state.threshold2;

  // ---- test hooks ----
  window.__viewerState = () => ({
    threshold: state.threshold, threshold2: state.threshold2, zSlice: state.zSlice,
    triangles: lastTriangles, cameraPos: camera.position.toArray(),
  });
  window.__setState = (partial) => {
    Object.assign(state, partial);
    rebuild();
    updateSlice();
  };
  window.__frameCamera = frameCamera;
  window.__viewerReady = true;
})();
