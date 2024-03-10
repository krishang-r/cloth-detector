import styled from "styled-components";

export const PinGrid = styled.div`
  width: 100%;
  height: 100%;
  display: grid;
  grid-template-columns: repeat(auto-fit, ${(props) => props.pinWidth}px);
  grid-auto-rows: 10px;
  column-gap: 1.25rem;
  justify-content: center;
`;
