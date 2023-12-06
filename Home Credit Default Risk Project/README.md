# Home Credit Default Risk

Dear reviewer/reader, please try to look at this readme file before moving on. 

- The plan for investigation, analysis, and POC building can be found here [Plan.md](https://github.com/TuringCollegeSubmissions/mfikad-ML.4/blob/master/Plan.md)

- There are two notebooks in this project one is [home_credit_risk.ipynb](https://github.com/TuringCollegeSubmissions/mfikad-ML.4/blob/master/notebooks/home_credit_risk.ipynb) which includes all EDA and Data Preprocessing, the other one is [home_credit_modelling.ipynb](https://github.com/TuringCollegeSubmissions/mfikad-ML.4/blob/master/notebooks/home_credit_modelling.ipynb) which includes the modelling part as well.

- The FastAPI backend I used for model deployment can be found in the [backend.py](https://github.com/TuringCollegeSubmissions/mfikad-ML.4/blob/master/backend.py) file.

- The endpoint for the API is https://home-credit.onrender.com

- The gradio web app can be found via this link [https://loan-prediction-051v.onrender.com/](https://home-credit-gradio.onrender.com) This might take some time to load since it is hosted on a free tier.

- The API I used which is deployed on the platform Render. The reason I used render and not GCP is because in our country credit card access is not available and I couldn't use the well known cloud providers.

- I have also confirmed that with Gierdrius that I can use other platforms.

![Screenshot 2023-08-25 132209](https://github.com/TuringCollegeSubmissions/mfikad-ML.3/assets/80324103/5d91f8fe-41de-4db3-8f6d-20da82d544cd)

- The entire code can be found inside [`app.py`](https://github.com/TuringCollegeSubmissions/mfikad-ML.4/blob/master/app.py) file, which uses Gradio.

- For the functions used in the notebook please refer the [`function.py`](https://github.com/TuringCollegeSubmissions/mfikad-ML.4/blob/master/notebooks/function.py) file.

## To run the app locally

First in the parent folder run this command to initialize the backend.

```console
foo@bar:~$ python -m app.py
```
