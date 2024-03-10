import { createGlobalStyle } from "styled-components";

export const GlobalStyle = createGlobalStyle`
    @import url('https://fonts.googleapis.com/css2?family=Open+Sans:wght@300;400;600;700&display=swap');

    body{
        font-family: 'Open Sans', sans-serif;
        padding: 0;
        margin: 0;
        box-sizing: border-box;
        background-color: #0B1623;
        color: #E7F6FF;
    }

    

    img{
        max-width: 100%;
        height: auto;
    }

    button{
        border: none;
    }
`