import axios from "axios";

const service = axios.create({
    baseURL: "https://stroke-xhw5.onrender.com",
    // baseURL: "http://127.0.0.1:8000"
});

export default service;