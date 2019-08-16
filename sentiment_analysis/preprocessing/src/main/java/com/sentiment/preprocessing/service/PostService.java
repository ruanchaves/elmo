package com.sentiment.preprocessing.service;

import com.sentiment.preprocessing.model.Post;
import java.util.List;

import com.sentiment.preprocessing.repository.PostRepository;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

@Service
public class PostService implements IPostService {

    @Autowired
    private PostRepository repository;

    @Override
    public List<Post> findAll() {
        List<Post> posts = (List<Post>) repository.findAll();

        return posts;
    }

}