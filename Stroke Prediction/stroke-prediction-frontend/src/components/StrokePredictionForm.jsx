import { useState } from "react";
import axios from "../utils/axios";
import "./StrokePredictionForm.css";

const genderOptions = [
  { label: "Female", value: "0" },
  { label: "Male", value: "1" },
  { label: "Other", value: "2" },
];

const marriedOptions = [
  { label: "Yes", value: "1" },
  { label: "No", value: "0" },
];

const workTypeOptions = [
  { label: "Private", value: "2" },
  { label: "Self-employed", value: "3" },
  { label: "Children", value: "4" },
  { label: "Govt_job", value: "0" },
  { label: "Never_worked", value: "1" },
];

const residenceOptions = [
  { label: "Urban", value: "1" },
  { label: "Rural", value: "0" },
];

const smokingStatusOptions = [
  { label: "Never smoked", value: "2" },
  { label: "Unknown", value: "0" },
  { label: "Formerly smoked", value: "1" },
  { label: "Smokes", value: "3" },
];

const hypertensionOptions = [
  { label: "Yes", value: "1" },
  { label: "No", value: "0" },
];

const heartDiseaseOptions = [
  { label: "Yes", value: "1" },
  { label: "No", value: "0" },
];

const StrokePredictionForm = () => {
  const [formData, setFormData] = useState({
    gender: "",
    age: "",
    hypertension: "",
    heart_disease: "",
    ever_married: "",
    work_type: "",
    Residence_type: "",
    avg_glucose_level: "",
    bmi: "",
    smoking_status: "",
  });
  const [prediction, setPrediction] = useState(null);
  const [isLoading, setIsLoading] = useState(false);

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData((prevFormData) => ({
      ...prevFormData,
      [name]: value,
    }));
  };

  const onSubmit = async (e) => {
    e.preventDefault();
    setIsLoading(true);
    try {
      const data = {
        ...formData,
        gender: Number(formData.gender),
        age: Number(formData.age),
        hypertension: Number(formData.hypertension),
        heart_disease: Number(formData.heart_disease),
        ever_married: Number(formData.ever_married),
        work_type: Number(formData.work_type),
        Residence_type: Number(formData.Residence_type),
        avg_glucose_level: Number(formData.avg_glucose_level),
        bmi: Number(formData.bmi),
        smoking_status: Number(formData.smoking_status),
      };

      const response = await axios.get("/predict", {
        params: data,
      });
      setPrediction(response.data);
      setIsLoading(false);
    } catch (error) {
      console.error("Error while fetching data:", error);
    }
  };

  return (
    <div className="form-container">
      <h2>Stroke Prediction Form</h2>
      <form onSubmit={onSubmit}>
        <div>
          <label>Gender:</label>
          <select name="gender" value={formData.gender} onChange={handleChange}>
            <option value="">Select Gender</option>
            {genderOptions.map((option) => (
              <option key={option.value} value={option.value}>
                {option.label}
              </option>
            ))}
          </select>
        </div>
        <div>
          <label>Ever Married:</label>
          <select
            name="ever_married"
            value={formData.ever_married}
            onChange={handleChange}
          >
            <option value="">Select Ever Married</option>
            {marriedOptions.map((option) => (
              <option key={option.value} value={option.value}>
                {option.label}
              </option>
            ))}
          </select>
        </div>
        <div>
          <label>Work Type:</label>
          <select
            name="work_type"
            value={formData.work_type}
            onChange={handleChange}
          >
            <option value="">Select Work Type</option>
            {workTypeOptions.map((option) => (
              <option key={option.value} value={option.value}>
                {option.label}
              </option>
            ))}
          </select>
        </div>
        <div>
          <label>Residence Type:</label>
          <select
            name="Residence_type"
            value={formData.Residence_type}
            onChange={handleChange}
          >
            <option value="">Select Residence Type</option>
            {residenceOptions.map((option) => (
              <option key={option.value} value={option.value}>
                {option.label}
              </option>
            ))}
          </select>
        </div>
        <div>
          <label>Smoking Status:</label>
          <select
            name="smoking_status"
            value={formData.smoking_status}
            onChange={handleChange}
          >
            <option value="">Select Smoking Status</option>
            {smokingStatusOptions.map((option) => (
              <option key={option.value} value={option.value}>
                {option.label}
              </option>
            ))}
          </select>
        </div>
        <div>
          <label>Hypertension:</label>
          <select
            name="hypertension"
            value={formData.hypertension}
            onChange={handleChange}
          >
            <option value="">Select Hypertension</option>
            {hypertensionOptions.map((option) => (
              <option key={option.value} value={option.value}>
                {option.label}
              </option>
            ))}
          </select>
        </div>
        <div>
          <label>Heart Disease:</label>
          <select
            name="heart_disease"
            value={formData.heart_disease}
            onChange={handleChange}
          >
            <option value="">Select Heart Disease</option>
            {heartDiseaseOptions.map((option) => (
              <option key={option.value} value={option.value}>
                {option.label}
              </option>
            ))}
          </select>
        </div>
        <div>
          <label>Age:</label>
          <input
            type="number"
            name="age"
            value={formData.age}
            onChange={handleChange}
          />
        </div>
        <div>
          <label>Average Glucose Level:</label>
          <input
            type="number"
            name="avg_glucose_level"
            value={formData.avg_glucose_level}
            onChange={handleChange}
          />
        </div>
        <div>
          <label>BMI:</label>
          <input
            type="number"
            name="bmi"
            value={formData.bmi}
            onChange={handleChange}
          />
        </div>
        <div>
          <button type="submit"> {isLoading ? "Loading... This is hosted on a free tier, so it may take a moment to load." : "Predict"}</button>
        </div>
      </form>
      {prediction !== null && (
        <div className="result-container">
          <h3>Result:</h3>
          <p>
            {prediction.stroke_prediction
              ? "Stroke predicted. 😷"
              : "No stroke predicted. 🎆"}
          </p>
          <p>Probability: {prediction.probability} %</p>
        </div>
      )}
    </div>
  );
};

export default StrokePredictionForm;
