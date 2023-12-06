import "./App.css";
import StrokePredictionForm from "./components/StrokePredictionForm";

function App() {
  return (
    <>
      <h1> Check if you will have a stroke 🧑‍⚕️ </h1>
      <div className="card">
        <StrokePredictionForm />

        <div className="footer">
          <p>
            The predictions made by this app are not 100% accurate. Please go to
            a real doctor if you are concerned about your health. Made with 💉
            by{" Minase "}
          </p>
        </div>
      </div>
    </>
  );
}

export default App;
