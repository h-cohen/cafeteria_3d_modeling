# Muon Tomography Detector – Technical Summary

## Detector Overview

This detector is a compact cosmic-ray muon tracking telescope designed for muon radiography and tomography.

The system consists of:

* **4 scintillator tracking planes**
* **23 triangular scintillator bars per plane**
* **92 instrumented channels total**
* Two orthogonal planes at the top
* Two orthogonal planes at the bottom

Typical detector dimensions:

| Parameter        | Value          |
| ---------------- | -------------- |
| Bars per layer   | 23             |
| Number of layers | 4              |
| Active area      | ~65 cm × 65 cm |
| Detector height  | ~80 cm         |
| Bar length       | 40 cm          |

The detector reconstructs the incoming and outgoing muon trajectory and measures muon flux as a function of angle.

---

# Geometry

## Layer Arrangement

```
Top X layer
Top Y layer

    Drift Space

Bottom X layer
Bottom Y layer
```

Planes are arranged as:

* X plane
* Y plane
* X plane
* Y plane

allowing full 3D track reconstruction.

Layer separation is adjustable and determines angular resolution.

Typical angular resolution:

* 15–40 mrad

depending on vertical spacing.

---

# Scintillator Bar

Each bar is an extruded triangular scintillator.

## Dimensions

| Parameter  | Value  |
| ---------- | ------ |
| Length     | 40 cm  |
| Base width | 3.3 cm |
| Height     | 1.7 cm |

Material:

* Polystyrene scintillator
* 1% PPO
* 0.03% POPOP
* TiO₂ reflective coating

---

# Optical Readout

Each bar contains:

* Central WLS fiber
* Saint-Gobain BCF-91A fiber
* Optical grease coupling
* One SiPM

SiPM model:

* SensL / OnSemi MicroJ 30035

### Important Design Detail

The final detector places approximately:

```
~0.5 m WLS fiber
between scintillator and SiPM
```

to suppress short attenuation-length effects and reduce DAQ dynamic range requirements.

---

# Readout Electronics

DAQ platform:

* CAEN DT5550W

Features:

* Citiroc ASIC frontend
* FPGA-based acquisition
* 14-bit ADC
* 80 MS/s sampling
* Programmable SiPM bias

Although each layer contains:

```
23 bars
```

the DAQ board provides:

```
32 channels
```

therefore:

```
9 channels unused per layer
```

(assuming one DAQ section per layer).

---

# Signal Model

Muon energy deposition is approximately proportional to path length inside the scintillator.

For each hit bar:

```
ADC integral
    ↓
Photoelectron estimate
    ↓
Deposited path length estimate
```

The reconstruction software generally uses calibrated integrated charge rather than waveform shape.

---

# Sub-Bar Position Reconstruction

The detector achieves sub-strip resolution using triangular scintillators.

A muon crossing two neighboring bars deposits charge in both.

Let:

* n = signal in first bar
* N = signal in second bar
* a = triangle width

Then:

x = a·n/(N+n)

where x is the crossing position relative to the bar edge.

This is the key reconstruction equation.

---

# Hit Reconstruction

Per layer:

1. Identify bars above threshold.
2. Accept:

   * one bar hit
   * two adjacent bar hits
3. Convert ADC → calibrated charge.
4. Estimate hit coordinate using charge sharing.
5. Produce:

```python
LayerHit:
    layer_id
    coordinate
    uncertainty
    charge
    timestamp
```

The orthogonal layer pair yields:

```python
(x, y, z)
```

for that detector station.

---

# Tracking

A valid muon track typically requires:

* hit in all 4 layers

Track fitting:

```python
x(z) = ax*z + bx
y(z) = ay*z + by
```

using straight-line least squares.

Outputs:

```python
Track:
    theta
    phi
    chi2
    hit_list
```

where:

* theta = zenith angle
* phi = azimuth

---

# Detector Resolution

Measured single-layer spatial resolution:

| Layer   | Resolution |
| ------- | ---------- |
| Typical | 3.5–4.5 mm |

Representative value:

```text
σ_hit ≈ 4 mm
```

This is substantially better than the physical bar pitch because of charge-sharing interpolation.

---

# Event Selection

Typical analysis cuts:

## Raw Event

At least one channel above threshold.

## Tracking Event

For every layer:

* ≥1 hit
* ≤2 adjacent hit bars

Reject:

* large clusters
* noisy events
* ambiguous topology

---

# Calibration

Per-channel gain calibration uses cosmic muons.

Procedure:

1. Integrate waveform.
2. Build charge spectrum.
3. Fit Landau distribution.
4. Extract MPV.
5. Normalize channel gains.

Store:

```python
ChannelCalibration:
    gain
    pedestal
    noise_sigma
    MPV
```

---

# Simulation Model

Reference simulation:

* GEANT4

Geometry:

* 4 detector layers
* 23 bars per layer
* realistic material definitions

Generated particles:

* cosmic-ray muons

Inputs:

```python
Muon:
    x
    y
    z
    px
    py
    pz
    energy
```

Detector response:

```python
signal ∝ path_length_in_bar
```

---

# Reconstruction Data Flow

```text
Raw ADC
    ↓
Pedestal subtraction
    ↓
Charge integration
    ↓
Thresholding
    ↓
Hit finding
    ↓
Charge sharing interpolation
    ↓
Track fitting
    ↓
Angular histogramming
    ↓
Muon flux map
    ↓
Tomographic inversion
```

---

# Tomography Products

The detector ultimately produces:

## 2D Flux Maps

```python
Flux(theta, phi)
```

## Transmission Maps

```python
Measured Flux
      /
Expected Flux
```

## Density-Length Maps

```python
ρL(theta, phi)
```

## 3D Tomographic Volume

Using multiple detector positions:

```python
VoxelDensity[x,y,z]
```

obtained via reconstruction methods such as:

* SIRT
* ART
* MLEM
* Bayesian inversion

---

# Recommended Core Software Objects

```python
Bar
Layer
Hit
Cluster
Track
Event
Calibration
DetectorGeometry
FluxMap
VoxelGrid
TomographyReconstruction
```

---

# Key Detector Numbers

| Quantity             | Value        |
| -------------------- | ------------ |
| Layers               | 4            |
| Bars/layer           | 23           |
| Total bars           | 92           |
| Bar length           | 40 cm        |
| Bar width            | 3.3 cm       |
| Bar height           | 1.7 cm       |
| Layer resolution     | ~4 mm        |
| Angular resolution   | 15–40 mrad   |
| Active area          | ~65 × 65 cm² |
| DAQ channels/layer   | 32           |
| Used channels/layer  | 23           |
| Spare channels/layer | 9            |

---

# Critical Analysis Assumptions

1. Muons travel in straight lines inside the detector.
2. Signal amplitude is proportional to scintillator path length.
3. Neighboring bar charge sharing enables sub-bar interpolation.
4. Detector alignment is known.
5. Gain calibration is stable over time.
6. Flux estimation is performed in angular bins.
7. Tomography relies on comparing measured flux to expected open-sky or reference flux.
8. Multiple detector positions are required for true 3D density reconstruction.

```
```

