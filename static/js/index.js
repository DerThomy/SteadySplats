document.addEventListener("DOMContentLoaded", (event) => {
    var b = document.querySelectorAll('.b-dics');
    b.forEach(element =>
        new Dics({
            container: element,
            textPosition: 'bottom',
            arrayBackgroundColorText: ['#000000', '#000000', '#000000'],
            arrayColorText: ['#FFFFFF', '#FFFFFF', '#FFFFFF'],
            linesColor: '#ffffff'
        })
    );

});

// Copy the BibTeX entry to the clipboard; where that is unavailable or denied, select it for manual copying
function copyBibtex(button) {
    var entry = document.getElementById('bibtex');
    var selectEntry = function () {
        window.getSelection().selectAllChildren(entry);
    };
    if (!navigator.clipboard) {
        selectEntry();
        return;
    }
    navigator.clipboard.writeText(entry.innerText).then(function () {
        button.innerText = 'Copied!';
        setTimeout(function () {
            button.innerText = 'Copy';
        }, 2000);
    }, selectEntry);
}
