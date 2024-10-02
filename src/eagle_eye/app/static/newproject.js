/**
 * Copyright 2024 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

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
