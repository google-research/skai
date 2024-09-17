const reopenProjectButton = document.getElementById('reopen_project');
const spinner = document.getElementById('loading-spinner');
reopenProjectButton.addEventListener('click', () => {
  reopenProjectButton.disabled = true;
  spinner.style.display = 'block';

  const projectId = reopenProjectButton.dataset.projectId;
  const imageMetadataPath = document.getElementById('imageMetadataPath').value;

  const url = `/project/${projectId}/reopen`;
  const xhr = new XMLHttpRequest();
  xhr.open('POST', url);
  xhr.setRequestHeader('Content-Type', 'application/json');
  xhr.onload = () => {
    if (xhr.status == 200) {
      window.location.href = `/project/${projectId}/summary`;
    } else if (xhr.status == 400) {
      alert(xhr.response);
      window.location.href = `/project/${projectId}/summary`;
    }
  };
  const requestBody = `{"imageMetadataPath": "${imageMetadataPath}"}`;
  xhr.send(requestBody);
});
