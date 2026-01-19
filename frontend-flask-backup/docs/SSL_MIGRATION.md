# SSL Certificate Configuration Update

## Summary of Changes

The BeautyAI Web UI has been updated to use Nginx as a reverse proxy with Let's Encrypt SSL certificates instead of self-signed certificates.

### What Changed

1. **Nginx Configuration**: Updated to use Let's Encrypt certificates at `/etc/letsencrypt/live/dev.gmai.sa/`
2. **Flask App**: Simplified to run in HTTP mode only (port 5000) behind Nginx proxy
3. **Self-signed Certificates**: Removed `cert.pem` and `key.pem` from the project as they're no longer needed
4. **Service Configuration**: Updated systemd service to set `BEHIND_PROXY=true`

### SSL Certificate Locations

- **Let's Encrypt certificates** (used by Nginx): `/etc/letsencrypt/live/dev.gmai.sa/`
  - `fullchain.pem` - Full certificate chain
  - `privkey.pem` - Private key
- **Self-signed certificates** (removed): `~/benchmark_and_test/src/web_ui/cert.pem` and `key.pem`

### Architecture

```
Internet → Nginx (HTTPS:443/8443) → Flask App (HTTP:5000)
```

- Nginx handles SSL termination and serves HTTPS
- Flask app runs in HTTP mode behind the proxy
- All SSL/TLS encryption is handled by Nginx with Let's Encrypt certificates

### Benefits

1. **Production-ready SSL**: Let's Encrypt certificates are trusted by browsers
2. **Automatic renewal**: Let's Encrypt certificates auto-renew
3. **Better performance**: Nginx handles SSL termination efficiently
4. **Simplified app**: Flask app focuses on business logic, not SSL management

### Deployment

Run the deployment script to apply all configurations:
```bash
./deploy-nginx.sh
```

This will:
- Deploy Nginx configuration
- Restart both Nginx and the BeautyAI service
- Enable services to start on boot
