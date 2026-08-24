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
  const fieldLayer = decodeVolume(V.data_b64); // uint8, 0..255 maps to [0, value_range[1]]
  // Optional full-room 3D voxel volume (same grid): swap it in for the iso-surfaces
  // and z-slice to inspect the whole voxel cloud instead of the thin ceiling layer.
  const fieldFull = V.volume_full_b64 ? decodeVolume(V.volume_full_b64) : null;
  let field = fieldLayer;                // active volume driving iso-surfaces + z-slice
  let quantScale = V.quant_scale;        // active byte->density scale for the colorbar
  // Optional model-free layer: mean per-detector backprojection of measured
  // opacity onto the ceiling plane, on the same xy grid (nx*ny uint8).
  const dataLayer = V.backproject_b64 ? decodeVolume(V.backproject_b64) : null;
  // Optional DIP-enhanced reconstruction layer (nx*ny uint8), same grid.
  const dipLayer = V.dip_b64 ? decodeVolume(V.dip_b64) : null;
  // Optional artifact-cleaned layer (boundary blobs removed, denoised, sharpened).
  const cleanLayer = V.clean_b64 ? decodeVolume(V.clean_b64) : null;
  // Optional DIP + artifact-cleaned combo (best beam positions + boundary removal).
  const dipcleanLayer = V.dipclean_b64 ? decodeVolume(V.dipclean_b64) : null;
  // Optional single-detector reconstruction layers (nx*ny uint8, same grid):
  // the SIRT+TV solve trained on ONE position only, for comparing one detector
  // vs the two-detector fusion.
  const pos0Layer = V.pos0_b64 ? decodeVolume(V.pos0_b64) : null;
  const pos1Layer = V.pos1_b64 ? decodeVolume(V.pos1_b64) : null;

  // ---- three.js scene ----
  const container = document.getElementById("viewport");
  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x101418);
  const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 1000);
  const renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true });
  renderer.setPixelRatio(window.devicePixelRatio || 1);
  container.appendChild(renderer.domElement);

  const controls = new THREE.OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;

  scene.add(new THREE.AmbientLight(0xffffff, 0.35));
  scene.add(new THREE.HemisphereLight(0xcfd8e6, 0x2a2f36, 0.55));
  const dir = new THREE.DirectionalLight(0xffffff, 0.8);
  dir.position.set(1, 1.5, 1);
  scene.add(dir);

  const spacing = V.spacing_m;
  const origin = V.origin_m; // world (x, y, z) with z vertical (ceiling direction)
  // Three.js/OrbitControls assume Y is up; the volume's vertical axis is world Z.
  // toThree(x, y, z) remaps world -> scene axes consistently everywhere below.
  // (x, z, -y) is a proper rotation; the earlier (x, z, y) swap was a
  // reflection that rendered the whole room mirror-imaged in y, which is why
  // the top view never matched the 2D plots' orientation.
  const toThree = (x, y, z) => new THREE.Vector3(x, z, -y);
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

  // Default the z-slice to the layer that actually holds mass, not the top of
  // the grid -- for a single-height layered run the ceiling is nowhere near nz-1.
  function argmaxLayer() {
    const sums = new Float64Array(nz);
    for (let x = 0; x < nx; x++) {
      for (let y = 0; y < ny; y++) {
        const base = (x * ny + y) * nz;
        for (let z = 0; z < nz; z++) sums[z] += field[base + z];
      }
    }
    let best = 0;
    for (let z = 1; z < nz; z++) if (sums[z] > sums[best]) best = z;
    return best;
  }

  // Layered/height-scan runs now embed their fitted layer across the actual
  // number of z-voxels spanning its physical thickness (see reconstruct.py),
  // so the isosurface is real 3D geometry (an extruded slab with beam-shaped
  // ridges), not a flat silhouette -- show it by default like any other run.
  // `thinLayer` is kept only to pick a more useful default camera angle.
  function activeLayerCount() {
    const sums = new Float64Array(nz);
    let total = 0;
    for (let x = 0; x < nx; x++) {
      for (let y = 0; y < ny; y++) {
        const base = (x * ny + y) * nz;
        for (let z = 0; z < nz; z++) { sums[z] += field[base + z]; total += field[base + z]; }
      }
    }
    if (total <= 0) return nz;
    let n = 0;
    for (let z = 0; z < nz; z++) if (sums[z] > 0.01 * total) n++;
    return n;
  }
  const thinLayer = activeLayerCount() <= Math.max(3, Math.round(nz * 0.05));

  const state = {
    threshold: (V.suggested_iso[0] || 20) / 255,
    threshold2: (V.suggested_iso[1] || 60) / 255,
    showLow: false,
    showHigh: false,
    zSlice: argmaxLayer(),
    clipBelow: false,
    surface: "recon", // "recon" | "data" (measured backprojection, if embedded)
    shaded: false, // terrain: false = flat colormap colors, true = lit relief
    showVoxels: false, // render the whole volume as a colored voxel point cloud
    voxelCut: (V.suggested_iso[0] || 20) / 255, // density cutoff for the voxel cloud
  };

  function rebuild() {
    if (meshLo) { group.remove(meshLo); meshLo.geometry.dispose(); meshLo.material.dispose(); }
    if (meshHi) { group.remove(meshHi); meshHi.geometry.dispose(); meshHi.material.dispose(); }
    // Marching cubes on this field is a speckled blob at any threshold -- the
    // reconstruction noise is close in amplitude to the real beam signal, and
    // the beam pattern only reads clearly through the height-relief terrain
    // below (same principle as the 2D slice: color/shape context over a whole
    // region reads far better than a hard isosurface boundary). Both iso
    // surfaces are opt-in tools for looking at the raw volumetric field, off
    // by default.
    if (state.showLow) {
      meshLo = buildMesh(state.threshold, 0x4fa3ff, 0.35);
      group.add(meshLo);
    } else {
      meshLo = null;
    }
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
    buildVoxels();
  }
  let lastTriangles = 0;

  // Full-3D voxel cloud: every voxel of the ACTIVE volume above the density cutoff
  // is drawn as a colour-coded point (viridis by density), across all z -- a true
  // 3D voxel view rather than a single slice. The z-slice slider + "clip volume
  // below slice" cut the cloud at any height. Faint by design (it is a diffuse,
  // limited-angle volume); raise the cutoff to keep only the densest structure.
  let voxelCloud = null;
  let lastVoxelCount = 0;
  function buildVoxels() {
    if (voxelCloud) {
      scene.remove(voxelCloud);
      voxelCloud.geometry.dispose();
      voxelCloud.material.dispose();
      voxelCloud = null;
    }
    lastVoxelCount = 0;
    if (!state.showVoxels) return;
    const lo = state.voxelCut * 255;
    const cutZ = origin[2] + state.zSlice * spacing; // world-z cut plane
    const positions = [], colors = [];
    for (let ix = 0; ix < nx; ix++) {
      for (let iy = 0; iy < ny; iy++) {
        const base = (ix * ny + iy) * nz;
        for (let iz = 0; iz < nz; iz++) {
          const v = field[base + iz];
          if (v < lo) continue;
          const wz = origin[2] + iz * spacing;
          if (state.clipBelow && wz < cutZ) continue; // cut everything below the plane
          const p = toThree(origin[0] + (ix + 0.5) * spacing, origin[1] + (iy + 0.5) * spacing, wz);
          positions.push(p.x, p.y, p.z);
          const c = viridis(v / 255);
          colors.push(c[0] / 255, c[1] / 255, c[2] / 255);
        }
      }
    }
    const geom = new THREE.BufferGeometry();
    geom.setAttribute("position", new THREE.Float32BufferAttribute(positions, 3));
    geom.setAttribute("color", new THREE.Float32BufferAttribute(colors, 3));
    const mat = new THREE.PointsMaterial({
      size: spacing * 1.6, vertexColors: true, transparent: true,
      opacity: 0.8, sizeAttenuation: true, depthWrite: false,
    });
    voxelCloud = new THREE.Points(geom, mat);
    scene.add(voxelCloud);
    lastVoxelCount = positions.length / 3;
  }

  // Height-relief terrain of the slab at zSlice: color AND height both encode
  // density (viridis LUT), so the beam pattern reads as raised, colored ridges
  // -- genuinely 3D, and just as legible as the 2D slice plot because it's
  // built the same way (per-pixel color from a local percentile-normalized
  // value), just given shape instead of staying flat. The relief height is a
  // visualization encoding, not the layer's real physical thickness (which is
  // shown separately via the iso-surface tools and the z-width metric).
  // Colour palettes for the density LUT (terrain, colorbar, voxels). Selectable in
  // the panel; grayscale is computed. viridis(t) always reads the ACTIVE one.
  const PALETTES = {
    viridis: [[68,1,84],[71,13,96],[72,24,106],[72,35,116],[71,46,124],[69,56,130],[66,65,134],[62,74,137],[58,84,140],[54,93,141],[50,101,142],[46,109,142],[43,117,142],[40,125,142],[37,132,142],[34,140,141],[31,148,140],[30,156,137],[32,163,134],[37,171,130],[46,179,124],[58,186,118],[72,193,110],[88,199,101],[108,205,90],[127,211,78],[147,215,65],[168,219,52],[192,223,37],[213,226,26],[234,229,26],[253,231,37]],
    inferno: [[0,0,4],[22,11,57],[66,10,104],[106,23,110],[147,38,103],[188,55,84],[221,81,58],[243,120,25],[252,165,10],[246,215,70],[252,255,164]],
    magma: [[0,0,4],[24,15,62],[68,15,118],[114,31,129],[158,47,127],[205,64,113],[240,96,93],[253,149,103],[254,197,145],[252,238,199],[252,253,191]],
    turbo: [[48,18,59],[64,84,200],[35,140,241],[27,193,207],[70,229,127],[157,241,44],[224,213,35],[253,150,32],[236,79,15],[176,20,4],[122,4,3]],
    grayscale: null, // computed below
  };
  let activeLUT = PALETTES.viridis;
  function viridis(t) {
    t = Math.max(0, Math.min(1, t));
    if (!activeLUT) return [t * 255, t * 255, t * 255]; // grayscale
    const n = activeLUT.length - 1;
    const i = Math.min(n - 1, Math.floor(t * n));
    const f = t * n - i;
    const a = activeLUT[i], b = activeLUT[i + 1];
    return [a[0] + f * (b[0] - a[0]), a[1] + f * (b[1] - a[1]), a[2] + f * (b[2] - a[2])];
  }
  // HUD colorbar: same LUT as the terrain, labeled with the physical density
  // values (byte / quant_scale) at the normalization endpoints, so the 3D view
  // is quantitatively readable like the 2D matplotlib slice.
  function updateColorbar(loByte, hiByte, scale, useData) {
    const canvas = document.getElementById("colorbar");
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    for (let px = 0; px < canvas.width; px++) {
      const [r, g, b] = viridis(px / (canvas.width - 1));
      ctx.fillStyle = `rgb(${r | 0},${g | 0},${b | 0})`;
      ctx.fillRect(px, 0, 1, canvas.height);
    }
    const s = scale || 1;
    document.getElementById("colorbar-lo").textContent = (loByte / s).toFixed(2);
    document.getElementById("colorbar-hi").textContent = (hiByte / s).toFixed(2);
    document.getElementById("colorbar-unit").textContent =
      useData ? "measured opacity -ln(T)" : "density (1/m)";
  }

  // A fixed, modest relief (not scaled to the footprint size) keeps the terrain
  // visually anchored to the room and detectors instead of floating overhead
  // as a detached mountain range.
  const RELIEF = Math.max(0.6, nz * spacing * 0.08);
  let terrainMesh = null;
  let sliceVisible = true;

  function updateSlice() {
    const iz = Math.max(0, Math.min(nz - 1, state.zSlice));
    if (terrainMesh) { scene.remove(terrainMesh); terrainMesh.geometry.dispose(); terrainMesh.material.dispose(); }

    // Terrain source: the reconstruction's z-slice, the measured-data
    // backprojection (model-free), or the DIP-enhanced layer -- the latter two
    // are 2D layers on the same xy grid, always at the fitted layer height.
    const useData = state.surface === "data" && dataLayer;
    const useDip = state.surface === "dip" && dipLayer;
    const useClean = state.surface === "clean" && cleanLayer;
    const useDipClean = state.surface === "dipclean" && dipcleanLayer;
    const useP0 = state.surface === "pos0" && pos0Layer;
    const useP1 = state.surface === "pos1" && pos1Layer;
    const layer2d = useData ? dataLayer : useDip ? dipLayer : useClean ? cleanLayer
      : useDipClean ? dipcleanLayer : useP0 ? pos0Layer : useP1 ? pos1Layer : null;
    const rawAt = (ix, iy) => layer2d ? layer2d[ix * ny + iy] : field[(ix * ny + iy) * nz + iz];

    // Robust two-sided normalization: [p30, p92] of the slice's NONZERO values
    // maps to [0, 1]. The low anchor keeps the color range off the empty floor;
    // the high anchor is deliberately at p92, NOT ~p99: the horizontal beams,
    // the y<-3 band and the grid edges are ~2x brighter than the vertical
    // beams, and with a p99-style vmax they hog the color range and leave the
    // (weaker but fully real, verified in beam_verify.png) vertical beams as
    // indistinct mid-purples -- at p92 all five beams read clearly and the
    // brightest structures simply saturate, exactly like vmax-clipping in the
    // matplotlib slice plots.
    const nzv = [];
    for (let x = 0; x < nx; x++) for (let y = 0; y < ny; y++) {
      const v = rawAt(x, y);
      if (v > 0) nzv.push(v);
    }
    nzv.sort((a, b) => a - b);
    // The DIP-enhanced layer is sharper and more bimodal than the plain recon:
    // its value distribution tops out lower (p85 ~ half-scale), so the recon's
    // [p35, p85] window clamps DIP's upper 15% to flat yellow and buries the
    // beams. A higher, wider window [p30, p97] spends the color range on DIP's
    // full tonal spread -- all five beams pop as distinct bright stripes.
    const bimodal = useDip || useClean || useDipClean;  // sharp/denoised layers top out lower -> wider window
    const pLo = bimodal ? 0.30 : 0.35;
    const pHi = bimodal ? 0.97 : 0.85;
    const vlo = nzv.length ? nzv[Math.floor(pLo * (nzv.length - 1))] : 0;
    const vhi = Math.max(nzv.length ? nzv[Math.floor(pHi * (nzv.length - 1))] : 1, vlo + 1);
    const norm = (raw) => Math.max(0, Math.min((raw - vlo) / (vhi - vlo), 1.0));
    // Each surface source stores byte values against its own scale (byte/scale =
    // physical density, or opacity for the measured backprojection).
    const surfScale = useData ? V.backproject_scale : useDip ? V.dip_scale : useClean ? V.clean_scale
      : useDipClean ? V.dipclean_scale : useP0 ? V.pos0_scale : useP1 ? V.pos1_scale : quantScale;
    updateColorbar(vlo, vhi, surfScale, useData);

    // Grid vertices + a "skirt": every perimeter vertex gets a twin dropped to
    // the layer base, so the terrain's silhouette is a solid wall instead of a
    // ragged floating sheet.
    const nPerim = 2 * nx + 2 * ny - 4;
    const positions = new Float32Array((nx * ny + nPerim) * 3);
    const colors = new Float32Array((nx * ny + nPerim) * 3);
    for (let ix = 0; ix < nx; ix++) {
      for (let iy = 0; iy < ny; iy++) {
        const v = norm(rawAt(ix, iy));
        const wx = origin[0] + (ix + 0.5) * spacing;
        const wy = origin[1] + (iy + 0.5) * spacing;
        // Beams protrude DOWNWARD from the ceiling slab toward the room (and the
        // floor-mounted detectors below), as real structural beams do -- so denser
        // material displaces the terrain down from the layer plane, not up.
        const wz = origin[2] + iz * spacing - v * RELIEF;
        const p = toThree(wx, wy, wz);
        const k = (ix * ny + iy) * 3;
        positions[k] = p.x; positions[k + 1] = p.y; positions[k + 2] = p.z;
        const [r, g, b] = viridis(v);
        colors[k] = r / 255; colors[k + 1] = g / 255; colors[k + 2] = b / 255;
      }
    }
    const indices = [];
    for (let ix = 0; ix < nx - 1; ix++) {
      for (let iy = 0; iy < ny - 1; iy++) {
        const a = ix * ny + iy, b = ix * ny + iy + 1, c = (ix + 1) * ny + iy, d = (ix + 1) * ny + iy + 1;
        indices.push(a, c, b, b, c, d);
      }
    }
    // Perimeter ring in walk order, then the skirt wall quads.
    const ring = [];
    for (let ix = 0; ix < nx; ix++) ring.push(ix * ny);
    for (let iy = 1; iy < ny; iy++) ring.push((nx - 1) * ny + iy);
    for (let ix = nx - 2; ix >= 0; ix--) ring.push(ix * ny + ny - 1);
    for (let iy = ny - 2; iy >= 1; iy--) ring.push(iy);
    ring.forEach((gi, j) => {
      const k = (nx * ny + j) * 3, g = gi * 3;
      const base = toThree(0, 0, origin[2] + iz * spacing); // only Y (vertical) matters
      positions[k] = positions[g]; positions[k + 1] = base.y; positions[k + 2] = positions[g + 2];
      colors[k] = colors[g] * 0.55; colors[k + 1] = colors[g + 1] * 0.55; colors[k + 2] = colors[g + 2] * 0.55;
    });
    for (let j = 0; j < ring.length; j++) {
      const jn = (j + 1) % ring.length;
      const t0 = ring[j], t1 = ring[jn], b0 = nx * ny + j, b1 = nx * ny + jn;
      indices.push(t0, b0, t1, t1, b0, b1);
    }
    const geom = new THREE.BufferGeometry();
    geom.setAttribute("position", new THREE.BufferAttribute(positions, 3));
    geom.setAttribute("color", new THREE.BufferAttribute(colors, 3));
    geom.setIndex(indices);
    geom.computeVertexNormals();
    // Unlit by default: PBR lighting shades slopes and desaturates the LUT, so
    // the same data reads duller here than in the matplotlib plots. Flat
    // colors give exact colormap fidelity (identical to the 2D images); the
    // "shaded relief" toggle brings the lit material back for shape reading.
    const mat = state.shaded
      ? new THREE.MeshStandardMaterial({ vertexColors: true, side: THREE.DoubleSide, roughness: 0.7 })
      : new THREE.MeshBasicMaterial({ vertexColors: true, side: THREE.DoubleSide });
    terrainMesh = new THREE.Mesh(geom, mat);
    terrainMesh.visible = sliceVisible;
    scene.add(terrainMesh);
    lastTerrainTriangles = indices.length / 3;
  }
  let lastTerrainTriangles = 0;

  // detector markers: a footprint box on the floor (world z = pose.z) + a
  // vertical stalk to the detector's top layer + a text label sprite.
  function makeLabel(text) {
    const canvas = document.createElement("canvas");
    canvas.width = 256; canvas.height = 64;
    const ctx = canvas.getContext("2d");
    ctx.fillStyle = "rgba(16,20,24,0.85)";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.font = "bold 40px sans-serif";
    ctx.fillStyle = "#ffe08a";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText(text, canvas.width / 2, canvas.height / 2);
    const tex = new THREE.CanvasTexture(canvas);
    const mat = new THREE.SpriteMaterial({ map: tex, depthTest: false });
    const sprite = new THREE.Sprite(mat);
    sprite.scale.set(diag * 0.09, diag * 0.0225, 1);
    return sprite;
  }

  (V.detectors || []).forEach((d) => {
    const w = d.aperture_m || 0.65, h = d.height_m || 0.8;
    const boxGeom = new THREE.BoxGeometry(w, h, w);
    const boxMat = new THREE.MeshStandardMaterial({ color: 0x2ecc71, roughness: 0.5 });
    const box = new THREE.Mesh(boxGeom, boxMat);
    const p = toThree(d.x, d.y, d.z || 0);
    box.position.set(p.x, p.y + h / 2, p.z);
    box.rotation.y = ((d.yaw_deg || 0) * Math.PI) / 180;
    scene.add(box);

    const edges = new THREE.LineSegments(
      new THREE.EdgesGeometry(boxGeom),
      new THREE.LineBasicMaterial({ color: 0x0d3d20 })
    );
    box.add(edges);

    const label = makeLabel(d.id || "det");
    label.position.set(p.x, p.y + h + diag * 0.02, p.z);
    scene.add(label);
  });

  // Verified beam guides: the beam positions measured model-free from the raw
  // per-detector data (muontomo.beams). Drawn as lines floating just above the
  // terrain so the real, verified structure is locatable at a glance even
  // where the reconstruction renders a beam faintly.
  const beamGroup = new THREE.Group();
  scene.add(beamGroup);
  (function drawVerifiedBeams() {
    const vb = V.verified_beams;
    if (!vb) return;
    const zTop = vb.z + RELIEF + 0.25;
    const mat = new THREE.LineBasicMaterial({ color: 0xff6b6b, transparent: true, opacity: 0.85 });
    const x0 = origin[0], x1 = origin[0] + nx * spacing;
    const y0 = origin[1], y1 = origin[1] + ny * spacing;
    (vb.x || []).forEach((bx) => {
      if (bx < x0 || bx > x1) return;
      const g = new THREE.BufferGeometry().setFromPoints([toThree(bx, y0, zTop), toThree(bx, y1, zTop)]);
      beamGroup.add(new THREE.Line(g, mat));
    });
    (vb.y || []).forEach((by) => {
      if (by < y0 || by > y1) return;
      const g = new THREE.BufferGeometry().setFromPoints([toThree(x0, by, zTop), toThree(x1, by, zTop)]);
      beamGroup.add(new THREE.Line(g, mat));
    });
  })();

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
  // Data-chosen ceiling height from parallax autofocus (muontomo.focus): the
  // trusted cross-validated height when available, else the fast quick-look.
  if (V.focus) {
    const f = V.focus, el = document.getElementById("hud-focus");
    if (el) {
      if (f.z_m != null)
        el.textContent =
          `auto-focus height: ${f.z_m.toFixed(2)} m (cross-validated)` +
          (f.quicklook_z_m != null ? `  |  quick-look ${f.quicklook_z_m} m` : "") +
          (f.solve_z_m != null ? `  |  solved at ${f.solve_z_m} m` : "");
      else if (f.quicklook_z_m != null)
        el.textContent =
          `auto-focus (quick-look): ${f.quicklook_z_m} m` +
          (f.solve_z_m != null ? `  |  solved at ${f.solve_z_m} m` : "");
    }
  }

  $("surface").addEventListener("change", (e) => { state.surface = e.target.value; updateSlice(); });
  if (!dipLayer) { const o = $("surface-dip-opt"); if (o) o.remove(); }
  if (!cleanLayer) { const o = $("surface-clean-opt"); if (o) o.remove(); }
  if (!dipcleanLayer) { const o = $("surface-dipclean-opt"); if (o) o.remove(); }
  if (!pos0Layer) { const o = $("surface-pos0-opt"); if (o) o.remove(); }
  if (!pos1Layer) { const o = $("surface-pos1-opt"); if (o) o.remove(); }
  if (!dataLayer) { const o = $("surface"); const d = [...o.options].find(x => x.value === "data"); if (d) d.remove(); }
  // Only the recon surface guaranteed -- hide the whole selector if nothing to compare against.
  if (!dataLayer && !dipLayer && !cleanLayer && !dipcleanLayer && !pos0Layer && !pos1Layer) $("surface-row").style.display = "none";
  // Full-room voxel volume: swap the active volume that the iso-surfaces and the
  // z-slice read, so the whole 3D voxel cloud can be inspected as another option.
  if (!fieldFull) { const r = $("fullvol-row"); if (r) r.style.display = "none"; }
  $("toggle-fullvol").addEventListener("change", (e) => {
    const on = e.target.checked && fieldFull;
    field = on ? fieldFull : fieldLayer;
    quantScale = on ? V.volume_full_scale : V.quant_scale;
    rebuild();
    updateSlice();
  });
  $("toggle-beams").addEventListener("change", (e) => { beamGroup.visible = e.target.checked; });
  $("toggle-voxels").addEventListener("change", (e) => { state.showVoxels = e.target.checked; buildVoxels(); });
  $("voxel-cut").addEventListener("input", (e) => { state.voxelCut = +e.target.value; if (state.showVoxels) buildVoxels(); });
  $("toggle-shaded").addEventListener("change", (e) => { state.shaded = e.target.checked; updateSlice(); });
  if (!V.verified_beams) $("beams-row").style.display = "none";
  $("thresh-lo").addEventListener("input", (e) => { state.threshold = +e.target.value; rebuild(); });
  $("thresh-hi").addEventListener("input", (e) => { state.threshold2 = +e.target.value; rebuild(); });
  $("toggle-lo").addEventListener("change", (e) => { state.showLow = e.target.checked; rebuild(); });
  $("toggle-hi").addEventListener("change", (e) => { state.showHigh = e.target.checked; rebuild(); });
  $("zslice").addEventListener("input", (e) => { state.zSlice = +e.target.value; updateSlice(); if (state.clipBelow) rebuild(); });
  $("toggle-slice").addEventListener("change", (e) => { sliceVisible = e.target.checked; if (terrainMesh) terrainMesh.visible = sliceVisible; });
  $("toggle-clip").addEventListener("change", (e) => { state.clipBelow = e.target.checked; rebuild(); });
  $("cam-iso").addEventListener("click", () => frameCamera("iso"));
  $("cam-top").addEventListener("click", () => frameCamera("top"));
  $("cam-front").addEventListener("click", () => frameCamera("front"));

  // Colour palette: swap the active density LUT and re-colour the terrain, colorbar
  // and voxel cloud (iso-surfaces keep their fixed low/high colours).
  $("palette").addEventListener("change", (e) => {
    activeLUT = PALETTES[e.target.value] !== undefined ? PALETTES[e.target.value] : PALETTES.viridis;
    updateSlice();
    buildVoxels();
  });

  // Collapse / expand the whole control panel for an unobstructed view.
  $("panel-toggle").addEventListener("click", () => {
    const collapsed = $("panel").classList.toggle("collapsed");
    $("panel-toggle").innerHTML = collapsed ? "show &#9656;" : "hide &#9662;";
  });

  // Save the current 3D view as a PNG (preserveDrawingBuffer makes the canvas
  // readable; render once first so the latest frame is captured).
  $("save-png").addEventListener("click", () => {
    renderer.render(scene, camera);
    const a = document.createElement("a");
    a.href = renderer.domElement.toDataURL("image/png");
    a.download = `${V.run || "viewer"}_${state.surface}${state.showVoxels ? "_voxels" : ""}.png`;
    document.body.appendChild(a);
    a.click();
    a.remove();
  });

  const zslider = $("zslice");
  zslider.max = nz - 1;
  zslider.value = state.zSlice;
  $("thresh-lo").value = state.threshold;
  $("thresh-hi").value = state.threshold2;
  $("voxel-cut").value = state.voxelCut;
  $("toggle-lo").checked = state.showLow;
  $("toggle-hi").checked = state.showHigh;

  // ---- test hooks ----
  window.__viewerState = () => ({
    threshold: state.threshold, threshold2: state.threshold2, zSlice: state.zSlice,
    triangles: lastTriangles, terrainTriangles: lastTerrainTriangles, cameraPos: camera.position.toArray(),
    voxelCount: lastVoxelCount, showVoxels: state.showVoxels,
    thinLayer, sliceVisible, surface: state.surface,
    hasDataLayer: !!dataLayer, hasDipLayer: !!dipLayer, hasCleanLayer: !!cleanLayer,
    hasDipCleanLayer: !!dipcleanLayer,
    hasPos0Layer: !!pos0Layer, hasPos1Layer: !!pos1Layer, hasFullVolume: !!fieldFull,
  });
  window.__setState = (partial) => {
    Object.assign(state, partial);
    rebuild();
    updateSlice();
  };
  window.__frameCamera = frameCamera;
  window.__viewerReady = true;
})();
