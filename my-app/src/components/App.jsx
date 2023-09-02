import React, {useState, useEffect} from 'react';

import '../css/App.css';
import { FileUpload } from './FileUpload';

import 'bootstrap/dist/css/bootstrap.min.css';



export default function App() {

  return (
    <div className="App">
      <div className="title-container">
        <h1>Drop an image of your favourite animal and find out if it's a cat or a dog!</h1>
      </div>
      <FileUpload />
    </div>
  );
}

