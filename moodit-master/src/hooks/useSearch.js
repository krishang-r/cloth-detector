import { useEffect, useState } from "react";

export default function useSearch(query, pageNumber) {
  const [isLoading, setIsLoading] = useState(true)
  const [error, setError] = useState(false)
  const [pins, setPins] = useState([])

  const accessKey = process.env.REACT_APP_ACCESSKEY;

  useEffect(() => {
    setPins([])
  },[query])

  useEffect(() => {
    setIsLoading(true)
    setError(false)
    
    let fullQuery = `https://api.unsplash.com/photos/random?client_id=${accessKey}&count=30`
      
    if (query)
    fullQuery = `https://api.unsplash.com/search/photos?client_id=${accessKey}&page=${pageNumber}&per_page=30&query=${query}`
      
    fetch(fullQuery)
    .then(res => res.json())
    .then(data => {
      data && setPins(prevPins => {
        return [...prevPins, ...(query ? data.results : data)]
      })
      setIsLoading(false)
    })
    .catch(error => {
      console.log(error)
      setError(true)
    })
  },[query, pageNumber])

  return {isLoading, error, pins}
}
