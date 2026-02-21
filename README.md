# SMRK-GUEGAP Probe - v1

Repository scaffold.
# SMRK-GUEGAP Probe - v1

**Spectral Probe for the Twistor–SMRK Framework for Yang–Mills Mass Gap**

---

## Overview

SMRK-GUEGAP Probe is a deterministic numerical framework designed to test whether
arithmetic SMRK-style operator families can simultaneously exhibit:

1. GUE-like bulk spectral statistics  
2. Stable low-energy spectral gaps under truncation scaling  

The framework emphasizes:

- Deterministic operator construction  
- Dense full diagonalization (v1 protocol)  
- Canonical JSON serialization  
- SHA256 hash commitment of all runs  
- Strict separation between exploration and confirmation  

This repository implements the v1 pre-registered scan protocol.

---

## Scientific Scope

This project does **not** claim a proof of the Yang–Mills mass gap.

Instead, it provides a controlled numerical laboratory for testing the coexistence of:

- Random-matrix universality (GUE class)
- Emergent spectral gap stability in deterministic arithmetic operators

All hypotheses are explicitly falsifiable.

---

## Installation

```bash
git clone https://github.com/101researchgroup/smrk-guegap-probe.git
cd smrk-guegap-probe
pip install -r requirements.txt
