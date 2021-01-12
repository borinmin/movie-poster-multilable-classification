## Movie Poster Classification
Hello everyone who see this!
In this module is not the road map of Computer Vision(CV), maybe in the future. Whatever, I would to share a simple  tasks that related to machine learning/image classification.

If you don't have idea about what is ``image classification`` that is not matter at this time.
We only aimed to figure out what is  the capability of CV. I'm not sure you have heard about Face Recognition, Object Detection, etc. So, here we would like to play around it.
### Implementation of this work 
We split this work into two parts `` data learning`` and `` inference``. ``Data learning or training data`` is task that we let algorithm to learn from dataset( [Movie Poster Images](https://www.cs.ccu.edu.tw/~wtchu/projects/MoviePoster/index.html)). After ``data learning`` is completed,we will get a learned model that storing the knowledge experience of movie poster. ``Inference`` is the task that will come to play after ``data learning`` have done to perform the real image classification.

- ### Data learning (Optional)
    - Requirements:
        - [Dataset of Movie Poster](https://www.cs.ccu.edu.tw/~wtchu/projects/MoviePoster/index.html) (include: train.csv and images)
        - Machine Specs (*):
            -  Linux OS(GPU GeForce RTX 2060 8GB+, RAM 40GB+).
        - Setup package (*):  
            - ``pip3 install -r quirements.txt``
        - Setup computing accelerator (Optional)
            - NVIDIA CUDA (Nvidia driver) 
            - CUDNN (Machine learning library for training model) to handle training computation on GPU
    - Run
    ``python3 train.py --csv train_partial.csv --img_dir path/Images/
    
- ### Inference
    - ``pip3 install -r quirements.txt`` (if not done yet)
    - run``python3 inference.py --img <img_path>`` to inference and let's see the output result.
    
### Example

``python3 inference.py --img samples/fast.jpg``

output:
```python
Possible Movie Genres are: 
        Genre: Comedy
        Genre: Crime
        Genre: Drama

```
This is image of ``fast.jpg``
![fast.jpg](samples/fast.jpg)

# Thank you! 