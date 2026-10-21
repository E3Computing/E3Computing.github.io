// Submit the contact form to Google Forms in the background and show
// the result inline. The spam token is only added here, so the form
// requires JS to submit successfully.
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

    // Spam guard: the Google Form rejects responses without this token.
    var data = new FormData(form);
    data.append("entry.610561625", "k7Rq2vLx9");

    fetch(form.action, {
      method: "POST",
      mode: "no-cors",
      body: data
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
