import React, { useState } from "react";
import axios from 'axios';

const Chat = () => {
    const [messages, setMessages] = useState([]);
    const [newMessage, setNewMessage] = useState('');

    const handleNewMessageChange = (event) => {
        setNewMessage(event.target.value);
    };

    const handleSendMessage = async () => {
        let userMessage = {user: 'User', text: newMessage};
        setMessages(oldMessages => [...oldMessages, userMessage]);
        setNewMessage('');

        try {
            const response = await axios.get(`http://localhost:8000/question/?message=${encodeURIComponent(newMessage)}`);

            let botMessage = {user: 'Bot', text: response.data.response};
            setMessages(oldMessages => [...oldMessages, botMessage]);
        } catch (error) {
            console.error("Error in sending message", error);
        }
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
