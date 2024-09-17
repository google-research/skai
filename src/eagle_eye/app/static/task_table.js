const nextPageButton = document.getElementById('next_page_button');
nextPageButton?.addEventListener('click', () => {
  window.location.href = pageLinks.next;
});
const previousPageButton = document.getElementById('previous_page_button');
previousPageButton?.addEventListener('click', () => {
  window.location.href = pageLinks.previous;
});

const labelsFilter = document.getElementById('labels-filter');

labelsFilter?.addEventListener('change', () => {
  window.location.href = pageLinks.withLabel + labelsFilter.value;
});