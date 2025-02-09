# Asterisk Installation and Configuration Guide

## Table of Contents
1. [Introduction](#introduction)
2. [System Requirements](#system-requirements)
3. [Installation Steps](#installation-steps)
4. [Configuring SIP Clients](#configuring-sip-clients)
5. [Editing Configuration Files](#editing-configuration-files)
   - [SIP Configuration (sip.conf)](#sip-configuration-sipconf)
   - [Dial Plan Configuration (extensions.conf)](#dial-plan-configuration-extensionsconf)
6. [Testing the Configuration](#testing-the-configuration)
7. [Troubleshooting](#troubleshooting)

---

## Introduction
Asterisk is an open-source framework for building communications applications such as IP PBX systems, VoIP gateways, and conference servers. This guide provides step-by-step instructions for installing and configuring Asterisk on Ubuntu.

---

## System Requirements
- **Operating System**: Ubuntu 20.04 or later
- **Dependencies**:
  - `build-essential`
  - `libxml2-dev`
  - `libsqlite3-dev`
  - `uuid-dev`
  - `libncurses5-dev`
  - `libssl-dev`
  - `libjansson-dev`
  - `wget`
  - `curl`

---

## Installation Steps
1. **Update the System**:
   ```bash
   sudo apt update && sudo apt upgrade
   ```

2. **Install Dependencies**:
   ```bash
   sudo apt install build-essential libxml2-dev libsqlite3-dev uuid-dev libncurses5-dev libssl-dev libjansson-dev wget curl
   ```

3. **Download Asterisk**:
   ```bash
   cd /usr/src
   sudo wget http://downloads.asterisk.org/pub/telephony/asterisk/asterisk-18-current.tar.gz
   sudo tar -xvzf asterisk-18-current.tar.gz
   cd asterisk-18.*
   ```

4. **Build and Install Asterisk**:
   ```bash
   sudo ./configure
   sudo make
   sudo make install
   sudo make samples
   sudo make config
   sudo ldconfig
   ```

5. **Enable and Start Asterisk Service**:
   ```bash
   sudo systemctl enable asterisk
   sudo systemctl start asterisk
   sudo systemctl status asterisk
   ```

---

## Configuring SIP Clients
Asterisk uses SIP (Session Initiation Protocol) for communication. Configuration involves modifying `sip.conf` and `extensions.conf`.

---

## Editing Configuration Files

### SIP Configuration (sip.conf)
Edit the SIP configuration file:
```bash
sudo nano /etc/asterisk/sip.conf
```

#put our sip.conf file here

Save and exit:
```bash
Ctrl+O, Enter, Ctrl+X
```

Restart Asterisk:
```bash
sudo systemctl restart asterisk
```

---

### Dial Plan Configuration (extensions.conf)
Edit the extensions configuration file:
```bash
sudo nano /etc/asterisk/extensions.conf
```
#put our extensions.conf file here

Save and exit:
```bash
Ctrl+O, Enter, Ctrl+X
```

Restart Asterisk:
```bash
sudo systemctl restart asterisk
```

---

## Testing the Configuration
1. Open two SIP clients (e.g., Zoiper or Linphone) and configure them with the SIP credentials defined in `sip.conf`.
2. Register extensions 7001 and 7002.
3. Dial 7002 from 7001 to test internal calling.
4. Dial 6000 to test the echo service.

---

## Troubleshooting
- **Check SIP Registration Status**:
  ```bash
  sudo asterisk -rvvv
  sip show peers
  ```
- **Check Logs**:
  ```bash
  sudo tail -f /var/log/asterisk/messages
  ```
- **Restart Asterisk Service**:
  ```bash
  sudo systemctl restart asterisk
  ```

---

## Conclusion
You have successfully installed and configured Asterisk with SIP clients and dial plans. For advanced features like IVR or integrating Voicemail, refer to the [official Asterisk documentation](https://wiki.asterisk.org/).


