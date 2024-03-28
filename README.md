# Submission of team "Artificially Natural HHU" for SemEval 2024 task 8
**Authors:** [Nele Mastracchio](https://slam.phil.hhu.de/authors/nele), [Wiebke Petersen](https://user.phil.hhu.de/~petersen/), [Anna Stein](https://ansost.com), Cornelia Genz, Hanxin Xia, Vittorio Ciccarelli

**Affiliation:** [Heinrich Heine University Düsseldorf](https://www.hhu.de/)

This repository provides the code and data for our solution for subtask A of shared task 8 of SemEval 2024 for classifying human- and machine-written texts in English across multiple domains. We propose a fusion model consisting of RoBERTa based pre-classifier and two MLPs that have been trained to correct the pre-classifier using linguistic features. Our model achieves an accuracy of 85\% and ranks 26th out of 141 participants. 

## Requirements
The code is written in Python 3.11. The required packages can be installed via pip using the provided `requirements.txt` file:
```bash
pip install -r requirements.txt
```

## Data
The data for the task can be downloaded from the [official Task repository](https://github.com/mbzuai-nlp/SemEval2024-task8). The data should be placed in the `data/` directory.

## Usage


## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
