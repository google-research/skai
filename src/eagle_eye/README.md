# Skai - EagleEye

EagleEye is a simple multi-user image labeling system for use with SKAI.

## Setup

### Set up Firebase
- Create a Firebase project
  - Visit https://console.firebase.google.com/ and click "Create a project".
  - Click "Add Firebase to Google Cloud Project".
  - Select your SKAI GCP Project.
  - Unselect "Enable Google Analytics for this project".
- Create Firestore database
  - Visit https://console.firebase.google.com/project/\[project-name\]/overview
  - In the left sidebar, click "All products".
  - Click "Cloud Firestore". "Firestore Database" should now appear under the
    "Project shortcuts" heading in the left sidebar.
  - Click "Create database".
  - Leave Database ID as "(default)". Select a location that is colocated with
    your SKAI data bucket. Click "Next".
  - Select "Start in production mode". Click "Create".
- Register Web App
  - Visit https://console.firebase.google.com/project/\[project-name\]/overview
  - Click "Add app".
  - Click "Web". The icon looks like "</>".
  - Enter "eagle-eye" for App nickname. Don't select the Firebase Hosting
    option. Click "Register app".
  - In the next screen, copy the data structure that looks like the following
    somewhere for later. Then click "Continue to console".

    ```
    const firebaseConfig = {
      apiKey: "abcdefghijklmnopqrstuvwxyzABCDE12345678",
      authDomain: "project-name.firebaseapp.com",
      projectId: "project-name",
      storageBucket: "project-name.appspot.com",
      messagingSenderId: "123456789",
      appId: "1:123456789012:web:1234567890abcdef123456"

    };
    ```

- Enable Firebase Authentication
  - https://console.firebase.google.com/project/\[your project\]/overview
  - In the left sidebar, click "All products".
  - Click "Authentication". "Authentication" should now appear under the
    "Project shortcuts" heading in the left sidebar.
  - In the tab bar under the "Authentication" header, click "Sign-in method".
  - Under "Additional providers", click "Google", then click "Enable", then
    "Save".
- Add users
  - Visit https://console.firebase.google.com/project/\[project-name\]/overview
  - In the left sidebar, click on the gear next to "Project Overview", then
    click "Users and permissions".
  - For each user you would like to access the labeling tool, click
    "Add member", enter their email address, and select the "Viewer" role.
    Note that the email address must work with Google OAuth, e.g. a gmail
    account or a Google Workspaces account.

### Create Service Account
- Follow these
  [instructions](https://cloud.google.com/iam/docs/service-account-overview)
  to create a service account to run EagleEye. The service account name can be
  "eagleeye" or anything you like. The service account needs these IAM roles:
  - Firebase Admin
  - Storage Object User
- Follow these
  [instructions](https://cloud.google.com/iam/docs/keys-create-delete)
  to create and download a service account JSON key file to your workstation.

### Deploy App Using Cloud Run
- On your workstation, create a text file named `config.json`. The contents of
  the file should be the `firebaseConfig` JSON data structure you copied
  previously. The file should look like this:

  ```
  {
    "firebaseConfig": {
      "apiKey": "abcdefghijklmnopqrstuvwxyzABCDE12345678",
      "authDomain": "project-name.firebaseapp.com",
      "projectId": "project-name",
      "storageBucket": "project-name.appspot.com",
      "messagingSenderId": "123456789",
      "appId": "1:123456789012:web:1234567890abcdef123456"
    }
  }
  ```

- In a terminal, run the following commands:

  ```
  $ export GOOGLE_APPLICATION_CREDENTIALS=[path to your JSON key file]
  $ cd skai/src/eagle_eye
  $ bash deploy.sh [path to your config.json]
  ```

- The output should contain lines similar to the following:

  ```
  Service [eagle_eye] revision [eagleeye-00001-kpb] has been deployed and is serving 100 percent of traffic.
  Service URL: https://eagleeye-1234567890.us-central1.run.app
  ```

- Copy the Service URL. That is the URL of the EagleEye app.

### Add Service URL to Authorized Domains
- Visit https://console.firebase.google.com/u/0/project/\[project-name\]/authentication/settings
- Click "Authorized domains".
- Click "Add domain", then paste the EagleEye service URL into the input box and
  click "Add".

### Allow Unauthenticated Domains
- Visit https://console.cloud.google.com/
- Select your project in the project dropdown.
- Open the navigation menu in the top left, then select "Cloud Run".
- You should see "eagle-eye" listed. Click on it.
- In the tab bar, click "SECURITY".
- Click "Allow unauthenticated invocations".

### Grant Admin Access
- In a terminal on your workstation, navigate to `skai/src/eagle_eye/admin`.
- Execute the following commands:

  ```
  $ python -m venv eagle-eye-admin
  $ source eagle-eye-admin/bin/activate
  $ pip install -r requirements.txt
  $ python grant_admin.py --email=[your email]
  ```
