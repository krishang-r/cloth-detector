import styled from "styled-components";

export const PageButton = styled.button`
  display: flex;
  background-color: rgba(255, 255, 255, 0.9);
  background-image: url(${props => props.background});
  background-position: 1em center;
  background-size: 1em;
  background-repeat: no-repeat;
  color: black;
  font-weight: bold;
  font-size: 0.8em;
  overflow: hidden;
  border: none;
  border-radius: 1.75em;
  padding: 1em 1em 1em 2.5em;
  cursor: pointer;
  height: 3em;
  margin: 0 0 0 1em;

  :hover{
    background-color: rgba(255, 255, 255, 1);
  }
`;
