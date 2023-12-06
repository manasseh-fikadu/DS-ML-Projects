# Stroke Prediction

Dear reviewer/reader, please try to look at this readme file before moving on.

- The web app can be found via this link https://stroke-backend.vercel.app

- The API I used which is deployed on the platform Render can be found here https://stroke-xhw5.onrender.com

- You can go to https://stroke-xhw5.onrender.com/docs to see the APIs endponits.

- The entire backend code can be found inside [`stroke_prediction_backend.py`](https://github.com/TuringCollegeSubmissions/mfikad-ML.2/blob/master/stroke_prediction_backend.py) file, which uses FastAPI.

- The notebook can be found inside the [`stroke.ipynb`](https://github.com/TuringCollegeSubmissions/mfikad-ML.2/blob/master/stroke.ipynb) file.

- For the functions used in the notebook please refer the [`functions.py`](https://github.com/TuringCollegeSubmissions/mfikad-ML.2/blob/master/functions.py) file.

- The entire frontend logic can be found in side the [`stroke_prediction_frontend`](https://github.com/TuringCollegeSubmissions/mfikad-ML.2/tree/master/stroke-prediction-frontend) folder. It is built using React.

## To run the app locally

First in the parent folder run this command to initialize the backend.

```console
foo@bar:~$ uvicorn stroke_prediction_backend:app --reload
```

After that to initialize the front end first move in to the folder

```console
foo@bar:~$ cd stroke_prediction_frontend
```

Once in there run the following command to install dependencies

```console
foo@bar:stroke_prediction_backend:~$ npm install
```

Now run the app using

```console
foo@bar:stroke_prediction_backend:~$ npm run dev
```

Have fun 🙂 Cheers 🥂
