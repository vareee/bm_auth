## ABOUT
This software is designed to use biometric authentication in the Linux OS family.  
To use it, you need to install the Python dependencies used in the code and make changes to the PAM configuration files.  
The file for editing is selected based on the needs of a specific user.

## USAGE
1. Record a voice sample:

    > python3 PATH_TO_FILE/ref_voice.py

2. Edit the PAM configuration file with the following line:
   
    > auth [required] pam_python.so PATH_TO_FILE/pam_voice_auth.py
      
    To implement the MFA, select the PAM flag "required". You can use the "sufficient" flag if you want to use only the voice authentication method, but this is not recommended from a security point of view.

3. Check the operation of the module by calling authentication
