# Test
<p align="center">
  
   <img src="https://github.com/mofadel/brain_tumor_detection/blob/main/images%20(1).jpeg"  width=72 height=72>
 

  <h3 align="center">MRI Brain Tumor Detection </h3>

  <!--- <p align="center">
    Project name is a <utility/tool/feature> that allows <insert_target_audience> to do <action/task_it_does>.
    <br>
    <a href="https://reponame/issues/new?template=bug.md">Report bug</a>
    ·
    <a href="https://reponame/issues/new?template=feature.md&labels=feature">Request feature</a>
  </p>
</p> --->

<!--- These are examples. See https://shields.io for others or to customize this set of shields. You might want to include dependencies, project status and licence info here --->

<!--- ![GitHub repo size](https://img.shields.io/github/repo-size/tkjohnson121/shrt)
![GitHub contributors](https://img.shields.io/github/contributors/tkjohnson121/shrt)
![GitHub stars](https://img.shields.io/github/stars/tkjohnson121/shrt?style=social)
![GitHub forks](https://img.shields.io/github/forks/tkjohnson121/shrt?style=social)
![Twitter Follow](https://img.shields.io/twitter/follow/tkjohnson121?style=social) --->

## Table of contents

- [Quick start](#quick-start)
- [Prerequisites](#prerequisites)
- [Dataset](#Dataset)
- [Usage](#Usage)
- [Contributing](#contributing)
- [Contributors](#contributors)
- [Contact](#contact)
- [License](#license)

## Quick start
Our project is a deep learning based image segmentation solution to first detect the brain tumor and predict the likelihood of tumor existence.
- To begin, clone the repo: 
``` 
git clone link_to_the_repo && cd folder_name 

``` 



## Prerequisites

Ensure you have met the following requirements by installing the following packages:

<!--- These are just example requirements. Add, duplicate or remove as required --->
  
* pandas==1.0.5
* numpy==1.18.5
* albumentations==0.5.2
* opencv_python_headless==4.4.0.46
* matplotlib==3.2.2
* torchvision==0.7.0
* torch==1.6.0
* scikit_learn==0.23.2

It can be done through this command line: ```pip install -r requirements```
## Dataset

The data can be found in this link: <a href="https://www.kaggle.com/mateuszbuda/lgg-mri-segmentation
">Brain MRI segmentation</a>. It can be loaded following these steps:

- Generate a key in you kaggle account.
- run these commands:

``` 
!mkdir ~/.kaggle/
!echo '{"username":"your_user_name","key":"key_you_have_generated"}' > ~/.kaggle/kaggle.json
!kaggle datasets download -d mateuszbuda/lgg-mri-segmentation

```
## How to Train & Test
This <a href="https://github.com/mofadel/brain_tumor_detection/blob/main/Brain-mri_RESTNET_UNET_MODEL.ipynb
">jupyter notebook</a> contains a simple tutorial on how to load the data, train and test the model.
Another alternative to train and test a model is to run the following script.
```
!python main.py
```
##### Please change $DATA_PATH variable at the very beginning of the notebook (if you are using it) to the path containing the MRI Brain Tumor detection data.
##### If you are running the script the $DATA_PATH variable can be located in the main.py file.


Add run commands and examples you think users will find useful. Provide
an options reference for bonus points!

## Contributing

<!--- If your README is long or you have some specific process or steps you want contributors to follow, consider creating a separate CONTRIBUTING.md file--->

To contribute to <project_name>, follow these steps:

1. Fork this repository.
2. Create a branch: `git checkout -b <branch_name>`.
3. Make your changes and commit them: `git commit -m '<commit_message>'`
4. Push to the original branch:
   `git push origin <project_name>/<location>`
5. Create the pull request.

Alternatively see the GitHub documentation on
[creating a pull request](https://help.github.com/en/github/collaborating-with-issues-and-pull-requests/creating-a-pull-request).

## Contributors

Thanks to the following people who have contributed to this project:

- [@Cyrille](https://github.com/) 📖
- [@Aissatou](https://github.com/sashakhaf) 📖

<!--- You might want to consider using something like the
[All Contributors](https://github.com/all-contributors/all-contributors)
specification and its
[emoji key](https://allcontributors.org/docs/en/emoji-key). --->

## Contact

If you want to contact me you can reach me at <fadelthior@gmail.com>.

## License

<!--- If you're not sure which open license to use see https://choosealicense.com/--->

This project uses the following license: [MIT licence file](https://github.com/mofadel/brain_tumor_detection/blob/main/LICENSE).

