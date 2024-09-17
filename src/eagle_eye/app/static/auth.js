import {initializeApp} from 'https://www.gstatic.com/firebasejs/10.12.2/firebase-app.js';
import {getAuth, GoogleAuthProvider, signInWithPopup} from 'https://www.gstatic.com/firebasejs/10.12.2/firebase-auth.js';

const redirectToHomepage =
    () => {
      window.location.replace('/projects');
    }

const sessionToken = document.cookie.split(';').find(
    (cookie) => cookie.startsWith('sessionToken='));
if (sessionToken == null) {
  const app = initializeApp(firebaseConfig);
  const provider = new GoogleAuthProvider();

  const auth = getAuth();
  signInWithPopup(auth, provider).then((result) => {
    return result.user.getIdToken().then(authToken => {
      const xhr = new XMLHttpRequest();
      xhr.open('POST', '/getSessionCookie');
      xhr.setRequestHeader('Content-Type', 'application/json');
      xhr.onload = (x) => {
        if (xhr.readyState == 4 && xhr.status == 200) {
          redirectToHomepage();
        }
      };
      const requestBody = `{"authToken": "${authToken}"}`;
      xhr.send(requestBody);
    });
  });
} else {
  redirectToHomepage();
}
