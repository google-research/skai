const nextTaskButton = document.getElementById('next_task');
nextTaskButton?.addEventListener('click', () => {
  const projectId = downloadCsvButton.dataset.project;
  window.location.href = `/project/${projectId}/next`;
});

const downloadCsvButton = document.getElementById('download_csv');
downloadCsvButton.addEventListener('click', () => {
  const projectId = downloadCsvButton.dataset.project;
  window.location.href = `/project/${projectId}/download_csv`;
});

const reopenProjectButton = document.getElementById('reopen_project');
reopenProjectButton?.addEventListener('click', () => {
  const projectId = reopenProjectButton.dataset.project;
  window.location.href = `/project/${projectId}/reopen`;
});

const dropTasksButton = document.getElementById('drop_tasks');
dropTasksButton?.addEventListener('click', () => {
  const projectId = dropTasksButton.dataset.project;

  dropTasksButton.disabled = true;
  const url = `/project/${projectId}/resetlabels`;
  const xhr = new XMLHttpRequest();
  xhr.open('POST', url);
  xhr.onload = () => {
    if (xhr.status == 200) {
      window.location.reload();
    }
  };
  xhr.send();
});

const deleteProjectButton = document.getElementById('delete_project');
const spinner = document.getElementById('loading-spinner');
deleteProjectButton?.addEventListener('click', () => {
  const projectId = dropTasksButton.dataset.project;

  deleteProjectButton.disabled = true;
  spinner.style.display = 'block';
  const url = `/project/${projectId}/delete`;
  const xhr = new XMLHttpRequest();
  xhr.open('POST', url);
  xhr.onload = () => {
    if (xhr.status == 200) {
      window.location.href = '/projects';
    }
  };
  xhr.send();
});

google.charts.load('current', {'packages': ['corechart']});
google.charts.setOnLoadCallback(drawLabelDistributionChart);

/**
 * Draw the label distribution chart and set the chart values.
 */
function drawLabelDistributionChart() {
  const chartData = google.visualization.arrayToDataTable([
    ['Category', 'Number of estimates'],
    ['Destroyed', labelDistributionData.destroyed_task_count],
    ['Major Damage', labelDistributionData.major_damage_task_count],
    ['Minor Damage', labelDistributionData.minor_damage_task_count],
    ['No Damage', labelDistributionData.no_damage_task_count],
    ['Bad Example', labelDistributionData.bad_example_task_count],
    ['I\'m Not Sure', labelDistributionData.not_sure_task_count],
  ]);

  const options = {'title': 'Damage Distribution'};
  const chart =
      new google.visualization.PieChart(document.getElementById('chart_div'));
  chart.draw(chartData, options);
}
