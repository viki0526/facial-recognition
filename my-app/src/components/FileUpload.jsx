import React, {useState, useEffect} from 'react'
import { Button } from 'react-bootstrap';

import '../css/FileUpload.css';

export function FileUpload () {
    const [file, setFile] = useState(null);
    const [text, setText] = useState('cat');
    const [resultOpacity, setResultOpacity] = useState(0)

    function handleFileChange (e) {
        setFile(e.target.files[0]);
        console.log(file);
    }

    function handleSubmit (e) {
        e.preventDefault()
        console.log(file);
        if (file) {
            const formData = new FormData();
            formData.append('image', file);

            fetch('http://127.0.0.1:5000/image', {
                method: 'POST',
                body: formData,
            })
            .then((response) => response.json())
            .then((data) => {
                // Handle the response from the Flask server
                console.log(data);
                setText(data.result)
                setResultOpacity(1)
            })
            .catch((error) => {
                // Handle any errors
                console.error('Error:', error);
            });
        } else {
            // Handle the case where no file is selected
            console.error('No file selected');
        }
    }

    return (
        <div className="file-upload-container">
            <form>
                <div class="mb-3">
                    <label for="formFile" className="form-label">Upload your image (jpg) here and click submit to find out</label>
                    <input className="form-control" type="file" id="formFile" accept=".png, .jpg" onChange={handleFileChange}></input>
                </div>
                <Button variant="primary" type="submit" value="Submit" onClick={handleSubmit}>Submit</Button>
            </form>
            <div className='result' style={{ opacity: resultOpacity }}>It's a {text}!</div>
        </div>
    )
}