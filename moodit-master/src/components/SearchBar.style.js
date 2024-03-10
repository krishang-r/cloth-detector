import styled from "styled-components";

export const SearchBar = styled.input`
    height: 3em;
    padding: 0.7em 0.7em 0.7em 2.6em;
    border-radius: 2em;
    border: 0px;
    background-color: #efefef;
    font-size: 1em;
    background-image: url(${require('../images/magnifying-glass.png')});
    background-size: 1em;
    background-repeat: no-repeat;
    background-position: center left 1em;
    display: flex;
    flex: 1;

    ::placeholder{
        color: #a5a5a5;
    }

    :hover{
        background-color: rgb(225 225 225)
    }

    :focus{
        outline: none;
        box-shadow: rgb(0 132 255 / 50%) 0px 0px 0px 4px;
        background-image: none;
        padding: 0.7em;
    }
`;