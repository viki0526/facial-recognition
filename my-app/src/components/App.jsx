import React, {useState, useEffect} from 'react';

import '../css/App.css';
import { FileUpload } from './FileUpload';

import 'bootstrap/dist/css/bootstrap.min.css';



export default function App() {
  const [data, setData] = useState({
    name: "",
    age: 0,
    date: "",
    programming: "",
  });

  useEffect(() => {
    // Using fetch to fetch the api from
    // flask server it will be redirected to proxy
    fetch("/data").then((res) =>
        res.json().then((data) => {
            // Setting a data from api
            setData({
                name: data.Name,
                age: data.Age,
                date: data.Date,
                programming: data.programming,
            });
        })
    );
  }, []);

  return (
    <div className="App">
      <div className="title-container">
        <h1>Drop an image of your favourite animal and find out if it's a cat or a dog!</h1>
      </div>
      <FileUpload />
    </div>
  );
}

