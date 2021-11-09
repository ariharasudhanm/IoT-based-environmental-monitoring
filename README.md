<div id="top"></div>
<!--


<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->

<div align="center">
  
[![Contributors][contributors-shield]](https://github.com/ariharasudhanm/IoT-based-environmental-monitoring/graphs/contributors)
[![Issues][issues-shield]](https://github.com/ariharasudhanm/IoT-based-environmental-monitoring/issues)
[![MIT License][license-shield]](https://github.com/ariharasudhanm/IoT-based-environmental-monitoring/blob/main/LICENSE)
[![LinkedIn][linkedin-shield]](https://www.linkedin.com/in/ariharasudhan/)
<!-- [![Forks][forks-shield]][forks-url] If needed add it later
[![Stargazers][stars-shield]][stars-url]  If needed add it later -->
 </p>
</div>
  
  
<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://github.com/ariharasudhanm/IoT-based-environmental-monitoring">
    <!-- <img src="images/logo.png" alt="Logo" width="80" height="80"> -->
  </a>
  <h3 align="center">IoT based environmental monitoring</h3>

  <p align="center">
    Development of environmental monitoring based on IoT cloud solutions using deep learning
    <br />
    <a href="https://github.com/ariharasudhanm/IoT-based-environmental-monitoring"><strong>Explore the docs »</strong></a>
    <br />
    <br />
    <!-- <a href="https://github.com/othneildrew/Best-README-Template">View Demo</a> -->
    ·
    <a href="https://github.com/ariharasudhanm/IoT-based-environmental-monitoring/issues">Report Bug</a>
    ·
    <a href="https://github.com/ariharasudhanm/IoT-based-environmental-monitoring/community">Request Feature</a>
  </p>
</div>



<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project

[![Humidity readings][product-screenshot]](https://github.com/ariharasudhanm/IoT-based-environmental-monitoring/blob/main/Images/Past_humidity_reading.png) |    [![Three-day humidity comparison][product-screenshot]](https://github.com/ariharasudhanm/IoT-based-environmental-monitoring/blob/main/Images/Three_day__humidity_data.png)

Initially DHT11 sensor is connected to Raspberry pi zero to mine the temperature and humidity data inside a closed room environment then data is then continuosly fed to thingspeak cloud using APIs. 

Project Overview:
* Data is fed from raspberry pi to cloud.
* Creating a machine learning model with the time series data to predict the future temperature and humidity of an environment.
* Model deployment in cloud.

For machine learning model initially, we will use neural networks to buid the model since its already one of the highly efficient for time series data.


<p align="right">(<a href="#top">back to top</a>)</p>



### Built With

These are the programming languages, libraries, frameworks, cloud services and other tools used in this project.

* [Python](https://www.python.org/)
* [Matlab](https://www.mathworks.com/)
* [Pytorch](https://pytorch.org/)
* [Numpy](https://numpy.org/)
* [Matplotlib](https://matplotlib.org/)
* [Thinspeak](https://thingspeak.com/)
* [Microsoft Azure cloud](https://azure.microsoft.com/en-us/)


<p align="right">(<a href="#top">back to top</a>)</p>



<!-- GETTING STARTED -->
## Getting Started



### Data Processing


<!-- USAGE EXAMPLES -->
## Model creation and deployment



<!-- ROADMAP -->
## Roadmap

- [x] Sensor setup using raspberry pi
- [x] Data feeding to cloud
- [x] Data visualization and pre-processing
- [] MOdel Creation (Ongoing)
- [] Evaluation and cloud deployment
    - [] Data improvement
    - [] Model deployment in different platforms

See the [open issues](https://github.com/ariharasudhanm/IoT-based-environmental-monitoring/issues) for a full list of proposed features (and known issues).

<p align="right">(<a href="#top">back to top</a>)</p>



<!-- CONTRIBUTING -->
## Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".
Don't forget to give the project a star! Thanks again!

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

<p align="right">(<a href="#top">back to top</a>)</p>



<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE.txt` for more information.

<p align="right">(<a href="#top">back to top</a>)</p>



<!-- CONTACT -->
## Contact

Your Name - [@AriharasudhanM](https://twitter.com/your_username) - ariharasudhan.muthusami@gmail.com

Project Link: [IoT-based-environmental-monitoring](https://github.com/ariharasudhanm/IoT-based-environmental-monitoring)

<p align="right">(<a href="#top">back to top</a>)</p>



<!-- ACKNOWLEDGMENTS -->
## Acknowledgments

Use this space to list resources you find helpful and would like to give credit to. I've included a few of my favorites to kick things off!

* [The future of environment monitoring](https://blog.smartsense.co/the-future-of-environmental-monitoring-deep-learning-and-artificial-intelligence)
* [An IoT based Environment Monitoring System](https://ieeexplore.ieee.org/document/9316050)
* [Deep Learning for Time Series Classification](https://github.com/hfawaz/dl-4-tsc)
* [REVIEW ON - IOT BASED ENVIRONMENT MONITORING SYSTEM](https://iaeme.com/MasterAdmin/Journal_uploads/IJECET/VOLUME_8_ISSUE_2/IJECET_08_02_014.pdf)
* [Advances in Smart Environment Monitoring Systems Using IoT and Sensors](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7309034/)
* [Design of an IoT based Real Time Environment Monitoring](https://www.matec-conferences.org/articles/matecconf/abs/2018/69/matecconf_cscc2018_03008/matecconf_cscc2018_03008.html)

<p align="right">(<a href="#top">back to top</a>)</p>



<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/othneildrew/Best-README-Template.svg?style=for-the-badge
[contributors-url]: https://github.com/othneildrew/Best-README-Template/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/othneildrew/Best-README-Template.svg?style=for-the-badge
[forks-url]: https://github.com/othneildrew/Best-README-Template/network/members
[stars-shield]: https://img.shields.io/github/stars/othneildrew/Best-README-Template.svg?style=for-the-badge
[stars-url]: https://github.com/othneildrew/Best-README-Template/stargazers
[issues-shield]: https://img.shields.io/github/issues/othneildrew/Best-README-Template.svg?style=for-the-badge
[issues-url]: https://github.com/othneildrew/Best-README-Template/issues
[license-shield]: https://img.shields.io/github/license/othneildrew/Best-README-Template.svg?style=for-the-badge
[license-url]: https://github.com/othneildrew/Best-README-Template/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/othneildrew
[product-screenshot]: images/screenshot.png
