# MosesChat

### Setup to run

To use Nginx as a reverse proxy for your FastAPI application, you can follow the steps below. 

Here, I'm assuming that your FastAPI application is running on the same machine as Nginx, and it's running on port 8000.

1. **Install Nginx:** 

You can install Nginx on Ubuntu by running:

```bash
sudo apt update
sudo apt install nginx
```

2. **Configure Nginx:**

You need to modify the Nginx configuration to act as a reverse proxy for your FastAPI application. First, back up the default configuration:

```bash
sudo cp /etc/nginx/sites-available/default /etc/nginx/sites-available/default.bak
```

Then, edit the Nginx configuration:

```bash
sudo nano /etc/nginx/sites-available/default
```

This will open the configuration file in a text editor. Replace the file's contents with the following configuration, which sets up Nginx as a reverse proxy to your FastAPI app running on port 8000:

```nginx
server {
    listen 80;
    server_name your_server_domain_or_IP;
    
    location / {
        proxy_pass http://localhost:3000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

Save and close the file when you're finished (in nano, you can do this by pressing `Ctrl+X`, then `Y` to confirm, then `Enter`).

3. **Check the configuration and restart Nginx:**

Before you apply the changes, check that there are no syntax errors in your configuration:

```bash
sudo nginx -t
```

If the configuration test is successful, restart Nginx to apply the changes:

```bash
sudo systemctl restart nginx
```

Now, you should be able to access your React application via your server's domain name or IP address (the one you put in place of `your_server_domain_or_IP` in the Nginx configuration) on port 80.

You might need to adjust firewall settings to allow HTTP traffic on port 80. On a Ubuntu system with `ufw` firewall, you can do this by running `sudo ufw allow 'Nginx Full'`.

Please note that the configuration above is a basic one. Depending on your requirements, you might need to adjust it. For instance, if you want to serve both HTTP and HTTPS traffic, you'll need a more complex configuration and an SSL certificate.


### To run

1. screen
2. npm start &


