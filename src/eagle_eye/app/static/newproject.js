const createProjectButton = document.getElementById('create_project');
const spinner = document.getElementById('loading-spinner');
createProjectButton.addEventListener('click', () => {
  createProjectButton.disabled = true;
  spinner.style.display = 'block';

  const projectName = document.getElementById('projectName').value;
  const imageMetadataPath = document.getElementById('imageMetadataPath').value;

  const url = `/projects/new`;
  const xhr = new XMLHttpRequest();
  xhr.open('POST', url);
  xhr.setRequestHeader('Content-Type', 'application/json');
  xhr.onload = () => {
    if (xhr.status == 200) {
      const newProjectId = JSON.parse(xhr.response)['projectId'];
      window.location.href = `/project/${newProjectId}/summary`;
    } else if (xhr.status == 400) {
      alert(xhr.response);
      window.location.href = '/projects';
    }
  };
  const requestBody = `{"projectName": "${
      projectName}", "imageMetadataPath": "${imageMetadataPath}"}`;
  xhr.send(requestBody);
});
