## ABOUT
This software is designed to use biometric authentication in the Linux OS family.  
To use it, you need to install the Python dependencies used in the code and make changes to the PAM configuration files.  
The file for editing is selected based on the needs of a specific user.

## USAGE
1. Create a symbolic link to python script:
    > sudo ln -s /usr/local/lib/x86_64-linux-gnu/bm_auth/bm_auth.py /usr/local/bin/bm_auth

2. Record a biometrics sample:

    > bm_auth [voice/face] add

3. List recorded face samples (you can not list recorded voice samples as it supposed to have the only one):
   
   > bm_auth face list
   
4. Remove a biometrics sample:
   
   > bm_auth [voice/face] remove [№]
   
5. Edit the PAM configuration file with the following line:
   
    > auth [required] pam_python.so PATH_TO_FILE/pam_face_auth.py
    > auth [required] pam_python.so PATH_TO_FILE/pam_voice_auth.py
      
    To implement the MFA, select the PAM flag "required". You can use the "sufficient" flag if you want to use only the voice authentication method, but this is not recommended from a security point of view.

6. Check the operation of the module by calling authentication
