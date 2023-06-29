import React, { useState } from "react";
import axios from 'axios';

const Chat = () => {
    const [messages, setMessages] = useState([]);
    const [newMessage, setNewMessage] = useState('');

    const handleNewMessageChange = (event) => {
        setNewMessage(event.target.value);
    };

    const handleSendMessage = async () => {
        setMessages([...messages, {user: 'User', text: newMessage}]);
        setNewMessage('');

        const response = await axios.get('http://localhost:8000/question/', {
            user_id: 'User',
            message: newMessage
        });
        setMessages([...messages, {user: 'Bot', text: response.data.response}]);
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

