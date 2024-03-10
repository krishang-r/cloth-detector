import styled from "styled-components";

export const Logo = styled.div`
    justify-content: center;
    align-items: center;
    border-radius: 50%;
    display: flex;
    cursor: pointer;
    width: 3em;
    height: 3em;
    padding: 0.7em;

    > *{
        width: 100%;
        height: 100%;
    }

    &:hover{
        background-color: rgb(0 0 0/0.06);
    }
`;