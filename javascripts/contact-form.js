// Submit the contact form to Google Forms in the background and show
// the result inline. Without JS, the form falls back to a normal POST.
document$.subscribe(function () {
  var form = document.getElementById("contact-form");
  if (!form) return;

  var button = form.querySelector("button[type=submit]");
  var status = form.querySelector(".contact-form__status");

  form.addEventListener("submit", function (event) {
    event.preventDefault();
    button.disabled = true;
    status.className = "contact-form__status";
    status.textContent = "Sending…";

    fetch(form.action, {
      method: "POST",
      mode: "no-cors",
      body: new FormData(form)
    })
      .then(function () {
        form.reset();
        status.classList.add("is-success");
        status.textContent = "Thank you! Your message has been sent.";
      })
      .catch(function () {
        status.classList.add("is-error");
        status.textContent = "Sorry, something went wrong. Please try again later.";
      })
      .finally(function () {
        button.disabled = false;
      });
  });
});
