# Fuzzy Extractors are Practical

A fuzzy key derivation system for irises with 91% TAR for 105 bits of entropy. This repository contains multiple submodules components which are used to build **ClearEyes** as described in [eprint:2024/100](https://eprint.iacr.org/2024/100).

## Repository Structure

This repository consists of three primary submodules:

- **feap-fe**: Neural network-based feature extraction for iris images
- **lsh-lock**: Analysis and evaluation of binary feature vectors for entropy and TAR
- **sample-lock**: A cryptographic implementation of sample-then-lock fuzzy extractor in C.

## Params
The params directory is unique to this repository. It contains the input-selection for `lsh-lock` and the zip file containing
subsets that were evaluated for performance in [eprint:2024/100](https://eprint.iacr.org/2024/100). These files need to be uncompressed. They were compressed only to get around github's upload limit.

For detailed build and usage instructions, refer to the README in each submodule.

## Citation

If you use this, please cite:

```
@misc{cryptoeprint:2024/100,
      author = {Sohaib Ahmad and Sixia Chen and Luke Demarest and Benjamin Fuller and Caleb Manicke and Alexander Russell and Amey Shukla},
      title = {Fuzzy Extractors are Practical: Cryptographic Strength Key Derivation from the Iris},
      howpublished = {Cryptology {ePrint} Archive, Paper 2024/100},
      year = {2024},
      url = {https://eprint.iacr.org/2024/100}
}
```