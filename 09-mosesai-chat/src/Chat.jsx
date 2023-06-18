import React, { useState } from "react";
import axios from 'axios';

function Chat() {
    const [messages, setMessages] = useState([]);
    const [input, setInput] = useState("");

    const sendMessage = async () => {
        setMessages([...messages, {text: input, sender: 'user'}]);
        const response = await axios.post('http://localhost:8000/generate-response/', { message: input });
        setMessages([...messages, {text: input, sender: 'user'}, {text: response.data.response, sender: 'bot'}]);
        setInput("");
    };

    return (
        <div>
            <div>
                {messages.map((message, index) => (
                    <p key={index} align={message.sender === 'user' ? "right" : "left"}>
                        <b>{message.sender}: </b>{message.text}
                    </p>
                ))}
            </div>
            <input
                value={input}
                onChange={(e) => setInput(e.target.value)}
            />
            <button onClick={sendMessage}>Send</button>
        </div>
    );
}

export default Chat;
