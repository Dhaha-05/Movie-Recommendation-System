

var btn=document.getElementById("button-submit");
btn.addEventListener("click",function validatePassword(event) {
    var password = document.getElementById("password").value;
    console.log(":"+password)
    var confirmPassword = document.getElementById("confirm_password").value;
    console.log(":"+confirmPassword)
    if (password != confirmPassword) {
        event.preventDefault()
        alert("Password and confirm password do not match.");
    }
})