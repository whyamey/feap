# Fuzzy Extractors are Practical

This repository implements ClearEyes, the current state-of-the-art fuzzy key derivation system for iris biometrics that achieves 92% True Accept Rate while maintaining 105 bits of entropy. The implementation accompanies our research paper [Fuzzy Extractors are Practical](https://eprint.iacr.org/2024/100).

All results were evaluated on the [IITD](https://www4.comp.polyu.edu.hk/~csajaykr/IITD/Database_Iris.htm) dataset. 

## What's Inside
The system combines three components:

- **feap-fe**: Feature extraction pipeline that transforms segmented iris images into binary feature vectors.
- **lsh-lock**: Analysis toolkit for zeta-sampling, quantifying entropy, and evaluating authentication performance
- **sample-lock**: Cryptographic implementation of the sample-then-lock fuzzy extractor written in C.

## Reproducing Our Results
The params directory contains the essential configuration files needed to exactly reproduce our paper's results:

- Subsets for lsh-lock through zeta-sampling.
- Split for confidence and test subsets. Do not use confidence subsets for testing. Split was obtained with random selection.
- Input selection of templates used for enrollment and authentication. 

In brief, to compute entropy and TAR:

```bash
cd lsh-lock

./target/release/lsh-lock analyze --templates /path/to/features/test/ --input /path/to/subsets/subset --dimensions 512 --count 200000
./target/release/lsh-lock tar-multi --templates /path/to/features/test --input /path/to/subsets/subset --count 200000 --dimensions 512 --base 3 --tries 5 --input-selection /path/to/base_3.json
```
For detailed build and usage instructions, see the README in each submodule.

## Citation
If you use our work, please cite:
```
@inproceedings{shukla2025fuzzy,
  title={Fuzzy Extractors are Practical: Cryptographic Strength Key Derivation from the Iris},
  author={Shukla, Amey and Demarest, Luke and Fuller, Benjamin and Ahmad, Sohaib and Manicke, Caleb and Russell, Alexander and Chen, Sixia},
  booktitle={Proceedings of the 2025 ACM SIGSAC Conference on Computer and Communications Security},
  pages={3605--3619},
  year={2025}
}
```

## License
This project is provided under GPL. The IITD dataset has its own licensing terms which must be respected.
