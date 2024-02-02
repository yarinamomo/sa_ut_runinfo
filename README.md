# Using Run-time Information to Enhance Static Analysis of Machine Learning Notebooks

This is the anonymized GitHub repository for our paper: **Using Run-time Information to Enhance Static Analysis of Machine Learning Notebooks**

### The *detailed* results can be found: [preliminary_results.pdf](https://anonymous.4open.science/r/sa_ut_runinfo-B605/preliminary_results.pdf)

## Dataset

- The original UT dataset can be found at https://github.com/ForeverZyh/TensorFlow-Program-Bugs [1]
- data4pythia: UT dataset with and without run-time information injected
- data4chatGPT: UT dataset with and without run-time information injected, with comments removed

An explanation of how the run-time information is injected can be found in integrate_runinfo.ipynb

## Static analyzers and usage

- Pythia [2], the explanation can be found in initial_experiments_GPT-4.ipynb
- GPT-4 [3], the explanation can be found in initial_experiments_Pythia.ipynb

## Results

- Overall results: preliminary_results.pdf
- Detailed results of GPT-4: outputs_chatGPT
- Detailed results of Pythia: outputs_pythia

## License

This project is licensed under the terms of the BSD 3-Clause License.

## References

[1] Yuhao Zhang, Yifan Chen, Shing-Chi Cheung, Yingfei Xiong, Lu Zhang, An Empirical Study on TensorFlow Program Bugs Proceedings of the 27th ACM SIGSOFT International Symposium on Software Testing and Analysis, 2018.

[2] Sifis Lagouvardos, Julian Dolby, Neville Grech, Anastasios Antoniadis, Yannis Smaragdakis, Static Analysis of Shape in TensorFlow Programs, 34th European Conference on Object-Oriented Programming, 2020.

[3] OpenAI et al. 2023. Gpt-4 technical report. (2023). arXiv: 2303.08774 [cs.CL].
