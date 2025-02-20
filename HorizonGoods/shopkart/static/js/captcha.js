// Function to generate a random CAPTCHA string
function generateCaptcha() {
    var chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789';
    var captcha = '';
    for (var i = 0; i < 6; i++) {
        var randomIndex = Math.floor(Math.random() * chars.length);
        captcha += chars[randomIndex];
    }
    $('#captcha').text(captcha); // Display the CAPTCHA string
    return captcha;
}

// Store the generated CAPTCHA value for comparison
var captchaCode = generateCaptcha();
console.log(captchaCode,"captchaCode")

// When the form is submitted
$('#login').click(function(event){
  
  
  // Get the user input
  var userCaptchaInput = $('#captcha_input').val();

  // Check if the entered CAPTCHA matches
  if (userCaptchaInput === captchaCode) {
      alert('CAPTCHA passed. Form submitted.');
      // Proceed with form submission or any other action you want to take
  } else {
    event.preventDefault()
      alert('Incorrect CAPTCHA. Please try again.');
      // Regenerate CAPTCHA code after failure
      captchaCode = generateCaptcha();
  }
})


function myFunction() {
  var x = $("#password");
  if (x.attr('type') === "password") {
    x.attr('type', 'text');
  } else {
    x.attr('type','password');
  }
}