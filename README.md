# Fuzzy Extractors are Practical

This repository implements ClearEyes, the current state-of-the-art fuzzy key derivation system for iris biometrics that achieves 91% True Accept Rate while maintaining 105 bits of entropy. The implementation accompanies our research paper [Fuzzy Extractors are Practical](https://eprint.iacr.org/2024/100).

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

./target/release/lsh-lock analyze --templates /path/to/features/test/ --input /path/to/subsets/subset --dimensions 512 --count 250000
./target/release/lsh-lock tar-multi --templates /path/to/features/test --input /path/to/subsets/subset --count 250000 --dimensions 512 --base 3 --tries 5 --input-selection /path/to/base_3.json
```
For detailed build and usage instructions, see the README in each submodule.

## Citation
If you use our work, please cite:
```
@misc{cryptoeprint:2024/100,
      author = {Sohaib Ahmad and Sixia Chen and Luke Demarest and Benjamin Fuller and Caleb Manicke and Alexander Russell and Amey Shukla},
      title = {Fuzzy Extractors are Practical: Cryptographic Strength Key Derivation from the Iris},
      howpublished = {Cryptology {ePrint} Archive, Paper 2024/100},
      year = {2024},
      url = {https://eprint.iacr.org/2024/100}
}
```

## License
This project is provided under GPL. The IITD dataset has its own licensing terms which must be respected.
