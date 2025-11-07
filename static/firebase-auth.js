window.TrueSightApp = {
    isUserLoggedIn: false,
    openModal: () => { console.error('Modal not initialized'); }
};

document.addEventListener('DOMContentLoaded', () => {

    const authModal = document.getElementById('auth-modal');
    const modalCloseBtn = document.getElementById('modal-close-btn');
    const modalTitle = document.getElementById('modal-title');
    const modalEmail = document.getElementById('modal-email');
    const modalPassword = document.getElementById('modal-password');
    const modalSubmitBtn = document.getElementById('modal-submit-btn');
    const modalToggleText = document.getElementById('modal-toggle-text');
    const modalToggleLink = document.getElementById('modal-toggle-link');
    const modalError = document.getElementById('modal-error');

    const authContainer = document.getElementById('auth-container');
    const signInBtn = document.getElementById('auth-signin-btn');
    const signUpBtn = document.getElementById('auth-signup-btn');
    const logoutBtn = document.getElementById('auth-logout-btn');

    let isSigningUp = false;

    function openModal(isSignUp = false) {
        isSigningUp = isSignUp;
        modalTitle.textContent = isSignUp ? 'Sign Up' : 'Sign In';
        modalSubmitBtn.textContent = isSignUp ? 'Sign Up' : 'Sign In';
        modalToggleText.textContent = isSignUp ? 'Already have an account?' : "Don't have an account?";
        modalToggleLink.textContent = isSignUp ? 'Sign In' : 'Sign Up';
        modalError.classList.add('hidden');
        modalError.textContent = '';
        authModal.classList.remove('hidden');
    }

    window.TrueSightApp.openModal = openModal;

    function closeModal() {
        authModal.classList.add('hidden');
        modalEmail.value = '';
        modalPassword.value = '';
    }

    signInBtn.addEventListener('click', (e) => {
        e.preventDefault();
        openModal(false);
    });

    signUpBtn.addEventListener('click', (e) => {
        e.preventDefault();
        openModal(true);
    });

    modalCloseBtn.addEventListener('click', closeModal);
    authModal.addEventListener('click', (e) => {
        if (e.target === authModal) closeModal();
    });

    modalToggleLink.addEventListener('click', (e) => {
        e.preventDefault();
        openModal(!isSigningUp);
    });
    
    function showModalError(message) {
        modalError.textContent = message;
        modalError.classList.remove('hidden');
    }

    modalSubmitBtn.addEventListener('click', (e) => {
        e.preventDefault();
        const email = modalEmail.value;
        const password = modalPassword.value;
        
        if (!email || !password) {
            showModalError('Please enter both email and password.');
            return;
        }

        modalError.classList.add('hidden');

        if (isSigningUp) {
            auth.createUserWithEmailAndPassword(email, password)
                .then((userCredential) => {
                    console.log('User signed up:', userCredential.user);
                    closeModal();
                })
                .catch((error) => {
                    showModalError(error.message);
                });
        } else {
            auth.signInWithEmailAndPassword(email, password)
                .then((userCredential) => {
                    console.log('User signed in:', userCredential.user);
                    closeModal();
                })
                .catch((error) => {
                    showModalError(error.message);
                });
        }
    });

    logoutBtn.addEventListener('click', (e) => {
        e.preventDefault();
        auth.signOut()
            .then(() => console.log('User signed out.'))
            .catch((error) => console.error('Sign out error:', error));
    });

    auth.onAuthStateChanged((user) => {
        
        window.TrueSightApp.isUserLoggedIn = !!user;

        if (user) {
            signInBtn.classList.add('hidden');
            signUpBtn.classList.add('hidden');
            logoutBtn.classList.remove('hidden');
        } else {
            signInBtn.classList.remove('hidden');
            signUpBtn.classList.remove('hidden');
            logoutBtn.classList.add('hidden');
        }
    });
});