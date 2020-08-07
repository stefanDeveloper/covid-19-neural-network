# COVID-19 CNN
Deep Vision Project IWR University Heidelberg

[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]

<!-- TABLE OF CONTENTS -->
## Table of Contents

* [About the Project](#about-the-project)
  * [Built With](#built-with)
* [Getting Started](#getting-started)
  * [Prerequisites](#prerequisites)
  * [Installation](#installation)
* [Roadmap](#roadmap)
* [License](#license)

<!-- ABOUT THE PROJECT -->
## About The Project

Project has been developed as final exam for Deep Vision. Detection of infected patients is an important task, expecially with the current Covid-19 pandemic. We want to explore the possibility of applying machine learning techniques to determine whether a patient is infected based on a X-Ray images of his lungs.

### Built With

* [PyTorch](https://pytorch.org/)
* [COVID-19 image data collection](https://github.com/ieee8023/covid-chestxray-dataset)
* [NIH](https://www.kaggle.com/nih-chest-xrays/data)
* [Torchxrayvision](https://github.com/mlmed/torchxrayvision)



<!-- GETTING STARTED -->
## Getting Started

To get a local copy up and running follow these simple steps.

### Prerequisites

First, you need to download the NIH dataset. You can use [aria2c](https://github.com/aria2/aria2).
* Use aria2c to download NIH dataset
```sh
aria2c NIH-e615d3aebce373f1dc8bd9d11064da55bdadede0.torrent
```
* Unzip NIH images
```sh
tar -xzf images-224.tar.gz
```
* Move NIH images
```sh
mv ./images-224 {PATH_TO_DATA_FOLDER}/data/images-nih
```
* Get COVD-19 images
```sh
git clone https://github.com/ieee8023/covid-chestxray-dataset.git
```
* Get COVD-19 images
```sh
mv ./images {PATH_TO_DATA_FOLDER}/data/images-covid
```
### Installation
 
1. Clone the repo
```sh
git clone https://github.com/stefanDeveloper/covid-19-neural-network.git
```
2. Run main
```sh
python main.py
```

<!-- ROADMAP -->
## Roadmap

See the [open issues](https://github.com/github_username/repo/issues) for a list of proposed features (and known issues).



<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE` for more information.

<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/stefanDeveloper/covid-19-neural-network.svg?style=flat-square
[contributors-url]: https://github.com/stefanDeveloper/covid-19-neural-network/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/stefanDeveloper/covid-19-neural-network.svg?style=flat-square
[forks-url]: https://github.com/stefanDeveloper/covid-19-neural-network/network/members
[stars-shield]: https://img.shields.io/github/stars/stefanDeveloper/covid-19-neural-network.svg?style=flat-square
[stars-url]: https://github.com/stefanDeveloper/covid-19-neural-network/stargazers
[issues-shield]: https://img.shields.io/github/issues/stefanDeveloper/covid-19-neural-network.svg?style=flat-square
[issues-url]: https://github.com/stefanDeveloper/covid-19-neural-network/issues
[license-shield]: https://img.shields.io/github/license/stefanDeveloper/covid-19-neural-network.svg?style=flat-square
[license-url]: https://github.com/stefanDeveloper/covid-19-neural-network/blob/master/LICENSE.txt
[product-screenshot]: images/screenshot.png
