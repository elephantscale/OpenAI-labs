import React, { useState } from "react";
import axios from 'axios';

const Chat = () => {
    const [messages, setMessages] = useState([]);
    const [newMessage, setNewMessage] = useState('');

    const handleNewMessageChange = (event) => {
        setNewMessage(event.target.value);
    };

    const handleSendMessage = async () => {
        const userMessage = { user: 'Learner', text: newMessage };
        setMessages((prevMessages) => [...prevMessages, userMessage]);

        try {
            const response = await axios.post('http://localhost:8000/ask', {
                question: newMessage
            });
            const botMessage = { user: 'MosesAI', text: response.data.answer };
            setMessages((prevMessages) => [...prevMessages, botMessage]);
        } catch (error) {
            console.log(error);
        }

        // Clear the message input field after sending the message
        setNewMessage('');
    };

    return (
        <div>
            <div>
                {messages.map((message, index) => (
                    <div key={index}>
                        <strong>{message.user}</strong>: {message.text}
                    </div>
                ))}
            </div>
            <input value={newMessage} onChange={handleNewMessageChange} />
            <button onClick={handleSendMessage}>Send</button>
        </div>
    );
};

export default Chat;
