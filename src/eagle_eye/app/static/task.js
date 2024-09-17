document.addEventListener('DOMContentLoaded', () => {
  const submitButton = document.getElementById('submit');
  const radioButtons = document.querySelectorAll('input');
  for (const radioButton of radioButtons) {
    if (radioButton.checked) {
      submitButton.disabled = false;
    }
    radioButton.addEventListener('click', () => {
      submitButton.disabled = false;
    });
  }
});

const inputButtons = document.getElementsByName('assessment');
const inputButtonsLength = inputButtons.length;
const inputButtonsHalfLength = Math.floor(inputButtons.length/2);

document.addEventListener(
  "keydown",
  (event) => {
    const keyName = event.key;

    if (keyName === "ArrowDown" && inputButtonsLength > 0) {
      const inputButton = inputButtons[inputButtonsHalfLength];
      inputButton.focus();
      inputButton.click();
    } else
    if (keyName === "ArrowLeft" && inputButtonsLength > 1) {
      const inputButton = inputButtons[inputButtonsHalfLength - 1];
      inputButton.focus();
      inputButton.click();
    }
    else if (keyName === "ArrowRight" && inputButtonsLength > 2) {
      const inputButton = inputButtons[inputButtonsHalfLength + 1];
      inputButton.focus();
      inputButton.click();
    } else {
    // Handle the case where the user presses a number key.
    try {
      const keyNumberValue = parseInt(keyName);
      if (keyNumberValue > 0 && keyNumberValue <= inputButtons.length) {
        const button = inputButtons[keyNumberValue - 1];
        button.focus();
        button.click();
        }
    } catch (e) {
      console.log(e);
    }
  }
  },
  false,
);