const createProjectButton = document.getElementById('create_project');
if (createProjectButton) {
  createProjectButton.addEventListener('click', () => {
    window.location.href = '/projects/new';
  });
}
