import React, {useState, useEffect} from 'react'
import { Button } from 'react-bootstrap';

export function FileUpload () {
    const [file, setFile] = useState(null)

    function handleFile (event) {
        setFile(event.target.files[0])
        console.log(event)
    }

    return (
        <div>
            <form>
                <input type="file" name="file" onChange={handleFile}></input>
                <Button variant="primary" type="submit" value="Submit">Submit</Button>
            </form>
        </div>
    )
}