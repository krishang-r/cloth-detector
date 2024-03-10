import styled from "styled-components";

export const DownloadButton = styled.div`
    justify-content: center;
    align-items: center;
    border-radius: 50%;
    display: flex;
    cursor: pointer;
    padding: 0.6em;
    width: 2.3em;
    height: 2.3em;
    background: rgba(255, 255, 255, 0.9);
    margin: 0 1em 1em auto;
    align-self: flex-end;

    > *{
        width: 100%;
        height: 100%;
    }

    &:hover{
    background-color: rgba(255, 255, 255, 1);
  }
`;